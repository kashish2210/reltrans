"""
Reltrans Professional Dashboard
Production-grade X-ray Reverberation Mapping Analysis Suite
Run: streamlit run reltrans_pro_dashboard.py
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import pandas as pd
import sys
import os
import io
import base64

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "reltrans" else SCRIPT_DIR
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# --- KERR METRIC & PHYSICS ---
PHYSICS_AVAILABLE = True

class KerrMetric:
    @staticmethod
    def disco(a):
        a = float(a)
        Z1 = 1 + (1 - a**2)**(1/3) * ((1+a)**(1/3) + (1-a)**(1/3))
        Z2 = np.sqrt(3*a**2 + Z1**2)
        if a >= 0: return 3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))
        else: return 3 + Z2 + np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))

    @staticmethod
    def dgsofac(a, h):
        Dh = h**2 - 2*h + a**2
        return np.sqrt(Dh / (h**2 + a**2))

    @staticmethod
    def dglpfacthick(r, a, h, mu=0.0):
        r = np.asarray(r, dtype=float)
        angvel = 1.0 / (r**1.5 + abs(a))
        Dh = h**2 - 2*h + a**2
        sindisk = np.sqrt(1.0 - mu**2)
        mathcalA = (r**2 + a**2)**2 - (r**2 - 2.0*r + a**2) * a**2 * sindisk
        Sig = r**2 + a**2 * mu**2
        gphiphi = mathcalA * sindisk**2 / Sig
        gsd = 1.0 - 2.0*r/Sig + 4.0*a*r*sindisk**2/Sig*angvel - gphiphi*angvel**2
        gsd = Dh / (h**2 + a**2) / gsd
        return np.sqrt(np.maximum(gsd, 1e-30))

    @staticmethod
    def dlgfacthick(a, mu0, alpha, r, mu=0.0):
        r = np.asarray(r, dtype=float)
        sin0 = np.sqrt(1.0 - mu0**2)
        sindisk = np.sqrt(1.0 - mu**2)
        angvel = 1.0 / (r**1.5 + abs(a))
        mathcalA = (r**2+a**2)**2 - (r**2-2.0*r+a**2)*a**2*sindisk
        Delta = r**2 - 2*r + a**2
        Sig = r**2 + a**2 * mu**2
        gtt = -(1.0 - 2.0*r / Sig)
        gtphi = -2.0*a*r*sindisk**2 / Sig
        gphiphi = mathcalA * sindisk**2 / Sig
        pt = -a*(alpha*sin0 + a*sindisk**2)
        pt = pt + (r**2+a**2)*(r**2+a**2+a*alpha*sin0)/Delta
        pt = pt / Sig
        pphi = -alpha*sin0/np.where(sindisk > 0, sindisk**2, 1e-30) - a
        pphi = pphi + a*(r**2+a**2+a*alpha*sin0)/Delta
        pphi = pphi / Sig
        num = np.sqrt(np.maximum(-gtt - 2.0*gtphi*angvel - gphiphi*angvel**2, 1e-30))
        den = -gtt*pt - gtphi*(pt*angvel+pphi) - gphiphi*pphi*angvel
        return num / np.where(np.abs(den) > 1e-30, den, 1e-30)

    @staticmethod
    def dlorfac(r, a):
        r = np.asarray(r, dtype=float)
        Delta = r**2 - 2*r + a**2
        BigA = (r**2+a**2)**2 - a**2*Delta
        Omega = 1.0 / (r**1.5 + abs(a))
        v = (Omega*BigA - 2*a*r) / (r**2 * np.sqrt(np.maximum(Delta, 1e-30)))
        return (1 - v**2)**(-0.5)

    @staticmethod
    def dareafac(r, a):
        r = np.asarray(r, dtype=float)
        Dm = r**2 - 2*r + a**2
        dArbydr = (r**4 + a**2*r**2 + 2*a**2*r) / Dm
        dArbydr = 2*np.pi*np.sqrt(np.maximum(dArbydr, 0))
        return KerrMetric.dlorfac(r, a) * dArbydr

class XillverTable:
    def __init__(self, fits_path):
        self.fits_path = fits_path
        self._loaded = False
        self._load()

    def _load(self):
        try:
            from astropy.io import fits as pyfits
            with pyfits.open(self.fits_path) as hdul:
                e = hdul['ENERGIES'].data
                self.energy_lo = e['ENERG_LO'].astype(np.float64)
                self.energy_hi = e['ENERG_HI'].astype(np.float64)
                self.energy_mid = 0.5 * (self.energy_lo + self.energy_hi)
                spec = hdul['SPECTRA']
                self.paramvals = spec.data['PARAMVAL'].astype(np.float64)
                self.spectra = spec.data['INTPSPEC'].astype(np.float64)
            self._loaded = True
        except ImportError:
            st.error("Missing `astropy` package. FITS tables (XILLVER) cannot be loaded. Run: `pip install astropy`")
            self._loaded = False

    def get_spectrum(self, Gamma=2.0, Afe=1.0, logxi=2.0, Ecut=300.0, incl=30.0):
        if not self._loaded: return np.array([]), np.array([]), np.array([])
        pv = self.paramvals
        dist = ((pv[:,0]-Gamma)/0.5)**2 + ((pv[:,1]-Afe)/1.0)**2 + \
               ((pv[:,2]-logxi)/1.0)**2 + ((pv[:,3]-Ecut)/100.0)**2 + \
               ((pv[:,4]-incl)/20.0)**2
        idx = np.argmin(dist)
        return self.energy_mid, self.spectra[idx], self.paramvals[idx]

def powerlaw_continuum(E, Gamma=2.0, Ecut=300.0):
    E = np.asarray(E, dtype=float)
    return E**(-Gamma) * np.exp(-E / Ecut)

class TransferFunction:
    def __init__(self, a=0.998, h=6.0, inc=30.0, rin=None, rout=400.0, nr=200):
        self.a = a; self.h = h; self.inc_deg = inc
        self.mu0 = np.cos(np.radians(inc))
        self.rin = rin if rin else KerrMetric.disco(a)
        self.rout = rout; self.nr = nr
        self._compute()

    def _compute(self):
        r = np.linspace(self.rin + 0.1, min(self.rout, 100), self.nr)
        self.radii = r
        self.gsd = KerrMetric.dglpfacthick(r, self.a, self.h)
        self.gdo = KerrMetric.dlgfacthick(self.a, self.mu0, 0.0, r)
        tau_sd = np.sqrt(self.h**2 + r**2) - self.h
        tau_do = r * np.abs(np.sin(np.radians(self.inc_deg))) * self.mu0
        self.tau = tau_sd + tau_do
        Gamma = 2.0
        area = KerrMetric.dareafac(r, self.a)
        self.emissivity = self.gsd**Gamma * area * (self.gdo)**3 / r

    def impulse_response(self, nt=500, tmax=None):
        if tmax is None: tmax = float(np.max(self.tau)) * 1.2
        t_grid = np.linspace(0, tmax, nt)
        dt = t_grid[1] - t_grid[0]
        psi = np.zeros(nt)
        for i, (tau_i, em_i) in enumerate(zip(self.tau, self.emissivity)):
            idx = int(tau_i / dt)
            if 0 <= idx < nt: psi[idx] += em_i
        from numpy import convolve
        kernel = np.ones(max(3, nt//50)) / max(3, nt//50)
        psi = convolve(psi, kernel, mode='same')
        psi = psi / (psi.max() + 1e-30)
        return t_grid, psi

    def lag_energy(self, E_grid, Gamma=2.0, Ecut=300.0, nu=0.01):
        nE = len(E_grid)
        lag = np.zeros(nE)
        for ie in range(nE):
            E_rest = E_grid[ie] / (self.gdo + 1e-30)
            fe_weight = 1.0 + 5.0 * np.exp(-0.5*((E_rest - 6.4)/0.3)**2)
            weights = self.emissivity * fe_weight * self.gdo**3
            weights = weights / (weights.sum() + 1e-30)
            lag[ie] = np.sum(self.tau * weights)
        return lag - np.median(lag)

@st.cache_data(show_spinner=False)
def cached_precompute_all(tables_dir=None):
    if tables_dir is None:
        tables_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
    data = {}
    a = 0.998; h = 6.0; inc = 30.0; Gamma = 2.0; Ecut = 300.0; logxi = 2.0
    data['params'] = dict(a=a, h=h, inc=inc, Gamma=Gamma, Ecut=Ecut, logxi=logxi)
    data['isco'] = KerrMetric.disco(a)
    r = np.linspace(data['isco'] + 0.1, 50, 200)
    data['r'] = r
    data['gsd'] = KerrMetric.dglpfacthick(r, a, h)
    data['gso'] = KerrMetric.dgsofac(a, h)
    data['gdo'] = KerrMetric.dlgfacthick(a, np.cos(np.radians(inc)), 0.0, r)
    data['lorentz'] = KerrMetric.dlorfac(r, a)
    data['area'] = KerrMetric.dareafac(r, a)
    E_cont = np.logspace(-0.5, 2.5, 500)
    data['E_cont'] = E_cont
    data['continuum'] = powerlaw_continuum(E_cont, Gamma, Ecut)
    
    fits_path = os.path.join(tables_dir, "xillver-a-Ec5_normalised.fits")
    if os.path.isfile(fits_path):
        xt = XillverTable(fits_path)
        E_refl, spec_refl, matched_params = xt.get_spectrum(Gamma=Gamma, Afe=1.0, logxi=logxi, Ecut=Ecut, incl=inc)
        data['E_refl'] = E_refl
        data['reflection'] = spec_refl
        data['refl_params'] = matched_params
        data['has_xillver'] = xt._loaded
    else:
        data['has_xillver'] = False

    tf = TransferFunction(a=a, h=h, inc=inc)
    t_grid, psi = tf.impulse_response(nt=400)
    data['t_tf'] = t_grid
    data['psi_tf'] = psi
    E_lag = np.logspace(-0.3, 1.5, 150)
    data['E_lag'] = E_lag
    data['lag'] = tf.lag_energy(E_lag, Gamma=Gamma, Ecut=Ecut)
    return data

# --- Import reltrans models (eager — load at startup) ---
MODELS_AVAILABLE = False
MODELS_LOAD_ERROR = ""
reltransDCp = reltransPL = reltransx = reltransDbl = rtdist = None

def load_models():
    global MODELS_AVAILABLE, MODELS_LOAD_ERROR, reltransDCp, reltransPL, reltransx, reltransDbl, rtdist
    if MODELS_AVAILABLE:
        return True

    import ctypes as ct
    type_float_p = ct.POINTER(ct.c_float)
    type_int_p = ct.POINTER(ct.c_int)

    # Try multiple library paths (Linux .so and macOS .dylib)
    lib_candidates = [
        os.path.join(SCRIPT_DIR, "lib_reltrans.so"),
        os.path.join(SCRIPT_DIR, "build", "lib", "libreltrans.so"),
        os.path.join(SCRIPT_DIR, "lib_reltrans.dylib"),
        os.path.join(SCRIPT_DIR, "build", "lib", "libreltrans.dylib"),
    ]

    lib = None
    errors = []
    for path in lib_candidates:
        if os.path.isfile(path):
            try:
                lib = ct.cdll.LoadLibrary(path)
                break
            except OSError as e:
                errors.append(f"{os.path.basename(path)}: {e}")
    
    if lib is None:
        found = [p for p in lib_candidates if os.path.isfile(p)]
        if not found:
            MODELS_LOAD_ERROR = f"No library found. Searched: {', '.join(os.path.basename(p) for p in lib_candidates)}"
        else:
            MODELS_LOAD_ERROR = "; ".join(errors)
        return False

    # Define wrapper functions directly (same as f2py_interface.py)
    def _make_wrapper(func_ptr):
        func_ptr.argtypes = [type_float_p, type_int_p, type_float_p, type_int_p, type_float_p]
        func_ptr.restype = None
        def wrapper(ear, params):
            import numpy as np
            ne = len(ear) - 1
            photar = np.zeros(ne, dtype=np.float32)
            func_ptr(
                ear.astype(np.float32).ctypes.data_as(type_float_p),
                ct.byref(ct.c_int(ne)),
                params.astype(np.float32).ctypes.data_as(type_float_p),
                ct.byref(ct.c_int(1)),
                photar.ctypes.data_as(type_float_p))
            return photar
        return wrapper

    try:
        reltransDCp = _make_wrapper(lib.tdreltransdcp_)
        reltransPL = _make_wrapper(lib.tdreltranspl_)
        reltransx = _make_wrapper(lib.tdreltransx_)
        reltransDbl = _make_wrapper(lib.tdreltransdbl_)
        rtdist = _make_wrapper(lib.tdrtdist_)
        MODELS_AVAILABLE = True
        MODELS_LOAD_ERROR = ""
    except AttributeError as e:
        MODELS_LOAD_ERROR = f"Symbol not found: {e}"
    except Exception as e:
        MODELS_LOAD_ERROR = f"Wrapper error: {e}"
    return MODELS_AVAILABLE

# Try loading at startup
load_models()

# ============================================================================
#  DEFAULT PATHS (auto-detect: works on both Windows and WSL/Linux)
# ============================================================================
_tables_candidates = [
    os.path.join(SCRIPT_DIR, "tables"),                      # reltrans/tables (relative)
    os.path.join(REPO_ROOT, "reltrans", "tables"),           # repo_root/reltrans/tables
    r"K:\Gsoc2026\reltrans\tables",                          # Windows absolute
    "/mnt/k/Gsoc2026/reltrans/tables",                       # WSL absolute
    os.environ.get("RELTRANS_TABLES", ""),                   # env variable
]
DEFAULT_TABLES_DIR = next((p for p in _tables_candidates if p and os.path.isdir(p)),
                           os.path.join(SCRIPT_DIR, "tables"))

# ============================================================================
#  STREAMLIT CONFIG
# ============================================================================
st.set_page_config(
    page_title="RELTRANS Professional Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'results' not in st.session_state:
    st.session_state.results = None
if 'phys_data' not in st.session_state:
    st.session_state.phys_data = None
if 'plot_history' not in st.session_state:
    st.session_state.plot_history = []
if 'history_idx' not in st.session_state:
    st.session_state.history_idx = -1
for key, val in [("chk_spectrum", True), ("chk_reflection", True), ("chk_continuum", False),
                 ("chk_gfactor", False), ("chk_transfer", False), ("chk_lag", False),
                 ("chk_emissivity", False), ("chk_ionization", False), ("chk_cumulative", False),
                 ("chk_all_plots", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================================
#  DARK RESEARCH THEME
# ============================================================================
st.markdown("""
<style>
    /* --- Dark observatory theme --- */
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label, section[data-testid="stSidebar"] span {
        color: #c9d1d9 !important;
    }
    h1, h2, h3, .main-header { color: #58a6ff !important; }
    .sub-header { color: #8b949e !important; font-size: 1.1rem; }
    .stMarkdown, p, label, span { color: #c9d1d9; }
    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetricLabel"] { color: #8b949e !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #161b22; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #21262d; border-radius: 6px; color: #8b949e;
        font-weight: 600; padding: 10px 20px; border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important; color: #fff !important; border-color: #1f6feb !important;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #30363d; color: #c9d1d9; }
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd); color: white;
        border: none; border-radius: 8px; font-weight: 600; padding: 0.6rem 1.5rem;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #388bfd, #58a6ff); }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #238636, #2ea043); color: white;
        border: none; border-radius: 6px; font-weight: 600;
    }
    .stNumberInput input, .stTextInput input, .stSelectbox > div > div {
        background-color: #0d1117 !important; color: #c9d1d9 !important;
        border: 1px solid #30363d !important; border-radius: 6px;
    }
    .stCheckbox label span { color: #c9d1d9 !important; }
    div[data-testid="stExpander"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
    .env-badge { display: inline-block; background: #238636; color: #fff; padding: 2px 8px;
        border-radius: 4px; font-size: 0.75rem; font-family: monospace; margin: 2px; }
    .env-badge-unset { background: #da3633; }
    .status-bar { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 12px 16px; margin: 8px 0; }
    code { color: #79c0ff !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
#  MATPLOTLIB DARK STYLE
# ============================================================================
PLOT_STYLE = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'legend.labelcolor': '#c9d1d9',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.formatter.use_mathtext': True,
}
COLORS = ['#58a6ff', '#f0883e', '#3fb950', '#d2a8ff', '#f778ba',
          '#a5d6ff', '#ffa657', '#7ee787', '#e2c5ff', '#ff9bce']

def apply_plot_style():
    plt.rcParams.update(PLOT_STYLE)

# ============================================================================
#  PLOT HELPER — makes individual figure, returns (fig, csv_data)
# ============================================================================
def fig_to_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def make_download_row(fig, csv_data, name):
    """Show download buttons for a single plot."""
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        st.download_button(f"Download {name}.png", fig_to_png_bytes(fig), f"{name}.png", "image/png", key=f"dl_png_{name}")
    with c2:
        if csv_data is not None:
            st.download_button(f"Download {name}.csv", csv_data, f"{name}.csv", "text/csv", key=f"dl_csv_{name}")

# ============================================================================
#  INDIVIDUAL PLOT GENERATORS — each returns (fig, csv_string)
# ============================================================================
def plot_energy_spectrum(E, flux, model_name, plot_type, log_scale):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_type == "E²F(E)":
        y = flux * E**2; ylabel = r"$E^2 \times F(E)$ [keV$^2$ ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$]"
    elif plot_type == "F(E)":
        y = flux; ylabel = r"$F(E)$ [ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$]"
    elif plot_type == "νF(ν)":
        y = flux * E; ylabel = r"$E \times F(E)$ [keV ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$]"
    else:
        y = flux / np.max(flux); ylabel = "Normalised Flux"
    if log_scale:
        ax.loglog(E, np.maximum(y, 1e-30), color=COLORS[0], lw=2, label=model_name)
    else:
        ax.plot(E, y, color=COLORS[0], lw=2, label=model_name)
    ax.set_xlabel("Energy (keV)", fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(f"Energy Spectrum — {model_name}", fontweight='bold', fontsize=13)
    ax.legend(framealpha=0.8); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"Energy_keV": E, "Flux": flux, plot_type: y}).to_csv(index=False)
    return fig, csv

def plot_reflection_spectrum(phys_data):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    if phys_data and phys_data.get('has_xillver'):
        E = phys_data['E_refl']; spec = phys_data['reflection']
        mask = (E > 0.3) & (E < 200) & (spec > 0); E, spec = E[mask], spec[mask]
        ax.loglog(E, spec, color=COLORS[1], lw=2, label="XILLVER FITS")
        fe = (E > 5.5) & (E < 7.5)
        if fe.any():
            ax.fill_between(E[fe], spec[fe].min()*0.5, spec[fe], alpha=0.25, color=COLORS[1])
            ax.axvline(6.4, color='#da3633', ls='--', lw=1, alpha=0.7, label='Fe Kα 6.4 keV')
        ax.set_title("XILLVER Reflection Spectrum (from FITS table)", fontweight='bold', fontsize=13)
    else:
        E = np.logspace(-0.3, 2, 500)
        spec = 0.5*E**(-0.8)*np.exp(-E/200) + 2*np.exp(-0.5*((E-6.4)/0.15)**2) + \
               0.8*np.exp(-0.5*((np.log10(E)-np.log10(25))/0.3)**2)
        ax.loglog(E, spec, color=COLORS[1], lw=2, label="Approximate")
        ax.set_title("XILLVER Reflection (approximation — no FITS table)", fontweight='bold', fontsize=13)
    ax.set_xlabel("Energy (keV)", fontweight='bold')
    ax.set_ylabel("Reflected Flux (norm.)", fontweight='bold')
    ax.legend(framealpha=0.8); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"Energy_keV": E, "Reflected_Flux": spec}).to_csv(index=False)
    return fig, csv

def plot_gfactor_profiles(phys_data):
    apply_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    r = phys_data['r']
    axes[0].plot(r, phys_data['gsd'], color=COLORS[0], lw=2)
    axes[0].set_xlabel("r (Rg)"); axes[0].set_ylabel(r"$g_{sd}$")
    axes[0].set_title("Source→Disk g-factor\n(dglpfacthick)", fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(r, phys_data['gdo'], color=COLORS[2], lw=2)
    axes[1].set_xlabel("r (Rg)"); axes[1].set_ylabel(r"$g_{do}$")
    axes[1].set_title("Disk→Observer g-factor\n(dlgfacthick)", fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(r, phys_data['lorentz'], color=COLORS[3], lw=2)
    axes[2].set_xlabel("r (Rg)"); axes[2].set_ylabel(r"$\gamma$")
    axes[2].set_title("Lorentz Factor\n(dlorfac)", fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"GR Factor Profiles  (a={phys_data['params']['a']}, h={phys_data['params']['h']} Rg)",
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    csv = pd.DataFrame({"r_Rg": r, "g_sd": phys_data['gsd'], "g_do": phys_data['gdo'],
                         "Lorentz": phys_data['lorentz']}).to_csv(index=False)
    return fig, csv

def plot_transfer_function(phys_data):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    t = phys_data['t_tf']; psi = phys_data['psi_tf']
    ax.plot(t, psi, color=COLORS[2], lw=2)
    ax.fill_between(t, 0, psi, alpha=0.15, color=COLORS[2])
    ax.set_xlabel("Time Delay (Rg/c)", fontweight='bold')
    ax.set_ylabel(r"$\Psi(t)$ (normalised)", fontweight='bold')
    p = phys_data['params']
    ax.set_title(f"Transfer Function  (a={p['a']}, h={p['h']} Rg, inc={p['inc']}°)", fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"t_Rgc": t, "Psi": psi}).to_csv(index=False)
    return fig, csv

def plot_lag_energy(phys_data):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    E = phys_data['E_lag']; lag = phys_data['lag']
    ax.semilogx(E, lag, color=COLORS[4], lw=2)
    ax.axhline(0, color='#30363d', lw=1)
    ax.set_xlabel("Energy (keV)", fontweight='bold')
    ax.set_ylabel("Lag (Rg/c)", fontweight='bold')
    ax.set_title("Lag-Energy Spectrum (from GR transfer function)", fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"Energy_keV": E, "Lag_Rgc": lag}).to_csv(index=False)
    return fig, csv

def plot_emissivity(phys_data):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    r = phys_data['r']; area = phys_data['area']
    gsd = phys_data['gsd']; Gamma = phys_data['params']['Gamma']
    emiss = gsd**Gamma * area / r
    ax.loglog(r, emiss / emiss.max(), color=COLORS[5], lw=2, label="Emissivity")
    ax.loglog(r, (r / r.min())**(-3), '--', color='#8b949e', lw=1, alpha=0.7, label=r"$r^{-3}$ (Newtonian)")
    ax.set_xlabel("r (Rg)", fontweight='bold')
    ax.set_ylabel("Emissivity (norm.)", fontweight='bold')
    ax.set_title("Disk Emissivity Profile", fontweight='bold', fontsize=13)
    ax.legend(framealpha=0.8); ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"r_Rg": r, "Emissivity_norm": emiss / emiss.max()}).to_csv(index=False)
    return fig, csv

def plot_ionization_profile(phys_data):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    r = phys_data['r']; gsd = phys_data['gsd']
    p = phys_data['params']
    # xi(r) ∝ gsd^Gamma / r^2 (simplified illumination)
    xi = gsd**p['Gamma'] / r**2
    xi = xi / xi.max() * 10**p['logxi']
    ax.loglog(r, xi, color=COLORS[6], lw=2)
    ax.set_xlabel("r (Rg)", fontweight='bold')
    ax.set_ylabel(r"$\xi(r)$ (erg cm s$^{-1}$)", fontweight='bold')
    ax.set_title("Radial Ionisation Profile", fontweight='bold', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"r_Rg": r, "xi": xi}).to_csv(index=False)
    return fig, csv

def plot_cumulative_flux(E, photar, log_scale):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    cumflux = np.cumsum(photar) / np.sum(photar)
    ax.plot(E, cumflux, color=COLORS[7], lw=2)
    ax.axhline(0.5, color='#da3633', ls='--', lw=1, alpha=0.7, label='50%')
    ax.set_xlabel("Energy (keV)", fontweight='bold')
    ax.set_ylabel("Cumulative Fraction", fontweight='bold')
    ax.set_title("Cumulative Flux Distribution", fontweight='bold', fontsize=13)
    if log_scale: ax.set_xscale('log')
    ax.legend(framealpha=0.8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"Energy_keV": E, "Cumulative_Fraction": cumflux}).to_csv(index=False)
    return fig, csv

def plot_continuum(phys_data):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    E = phys_data['E_cont']; flux = phys_data['continuum']
    p = phys_data['params']
    ax.loglog(E, flux, color=COLORS[0], lw=2)
    ax.set_xlabel("Energy (keV)", fontweight='bold')
    ax.set_ylabel(r"$F(E)$ (norm.)", fontweight='bold')
    ax.set_title(f"Power-Law Continuum  Γ={p['Gamma']}  Ecut={p['Ecut']} keV", fontweight='bold', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    csv = pd.DataFrame({"Energy_keV": E, "Continuum_Flux": flux}).to_csv(index=False)
    return fig, csv

# ============================================================================
#  COMBINED PLOT — selected plots in subplots
# ============================================================================
def make_combined_figure(plots_dict):
    """Take dict of {name: (fig, csv)} and combine into one multi-panel figure."""
    apply_plot_style()
    n = len(plots_dict)
    if n == 0: return None, None
    ncols = min(3, n); nrows = (n + ncols - 1) // ncols
    combined = plt.figure(figsize=(7 * ncols, 5 * nrows))
    for idx, (name, (orig_fig, _)) in enumerate(plots_dict.items()):
        # Copy axes from original to combined
        orig_ax = orig_fig.axes[0] if orig_fig.axes else None
        if orig_ax is None: continue
        ax = combined.add_subplot(nrows, ncols, idx + 1)
        for line in orig_ax.get_lines():
            ax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                    lw=line.get_linewidth(), ls=line.get_linestyle(), alpha=line.get_alpha() or 1.0,
                    label=line.get_label() if not line.get_label().startswith('_') else None)
        for coll in orig_ax.collections:
            try:
                paths = coll.get_paths()
                fc = coll.get_facecolor()
                if len(paths) > 0:
                    ax.fill_between([], [], alpha=0.15)  # placeholder
            except: pass
        ax.set_xlabel(orig_ax.get_xlabel())
        ax.set_ylabel(orig_ax.get_ylabel())
        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.set_xscale(orig_ax.get_xscale())
        ax.set_yscale(orig_ax.get_yscale())
        ax.grid(True, alpha=0.3)
        if orig_ax.get_legend(): ax.legend(fontsize=8, framealpha=0.8)
    combined.tight_layout()
    return combined, fig_to_png_bytes(combined)

# ============================================================================
#  HEADER
# ============================================================================
st.markdown('<h1 style="margin-bottom:0">RELTRANS Analysis Suite</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Production-grade X-ray Reverberation Mapping for Research</p>', unsafe_allow_html=True)

# Status bar
status_parts = []
status_parts.append(f'<span class="env-badge">Physics: {"[OK]" if PHYSICS_AVAILABLE else "[FAIL]"}</span>')
fortran_label = "[OK] Loaded" if MODELS_AVAILABLE else f"[FAIL] {MODELS_LOAD_ERROR}" if MODELS_LOAD_ERROR else "not loaded"
status_parts.append(f'<span class="env-badge {"" if MODELS_AVAILABLE else "env-badge-unset"}">Fortran: {fortran_label}</span>')
status_parts.append(f'<span class="env-badge">Tables: {DEFAULT_TABLES_DIR}</span>')
st.markdown(f'<div class="status-bar">{"  ".join(status_parts)}</div>', unsafe_allow_html=True)

# Session state
if 'results' not in st.session_state: st.session_state.results = {}
if 'phys_data' not in st.session_state: st.session_state.phys_data = None

# ============================================================================
#  SIDEBAR — Model, Env Vars, Tables
# ============================================================================
st.sidebar.markdown("## Configuration")
model_type = st.sidebar.selectbox("Model Flavour", [
    "reltransDCp (nthComp)", "reltransPL (Power-law)", "reltransx (ReflionX)",
    "reltransDbl (Double LP)", "rtdist (Distance-based)"],
    help="Choose RELTRANS model variant")

model_config = {
    "reltransDCp (nthComp)": ("tdreltransDCp", "reltransDCp", 21),
    "reltransPL (Power-law)": ("tdreltransPL", "reltransPL", 20),
    "reltransx (ReflionX)": ("tdreltransx", "reltransx", 21),
    "reltransDbl (Double LP)": ("tdreltransDbl", "reltransDbl", 27),
    "rtdist (Distance-based)": ("tdrtdist", "rtdist", 25),
}
model_name, model_func_name, n_params = model_config[model_type]

# --- Environment Variables ---
st.sidebar.markdown("---")
st.sidebar.markdown("## Environment Variables")

with st.sidebar.expander("RELTRANS_TABLES", expanded=True):
    tables_dir = st.text_input("Tables directory", value=DEFAULT_TABLES_DIR, key="tables_dir",
                                help="Path to directory containing normalised XILLVER FITS tables")
    available_tables = []
    if os.path.isdir(tables_dir):
        available_tables = [f for f in os.listdir(tables_dir) if f.endswith('.fits')]
        if available_tables:
            st.success(f"Found {len(available_tables)} FITS tables")
            selected_table = st.selectbox("Active table", available_tables,
                                           index=available_tables.index("xillver-a-Ec5_normalised.fits")
                                           if "xillver-a-Ec5_normalised.fits" in available_tables else 0)
        else:
            st.warning("No .fits files found")
            selected_table = None
    else:
        st.error("Directory not found")
        selected_table = None
    if st.button("Apply tables path", key="apply_tables"):
        os.environ["RELTRANS_TABLES"] = tables_dir
        st.success(f"Set RELTRANS_TABLES={tables_dir}")

with st.sidebar.expander("Physics Zones"):
    ion_zones = st.number_input("ION_ZONES", 1, 100, int(os.environ.get("ION_ZONES", "20")),
                                 help="Number of radial ionisation zones (default 20)")
    mu_zones = st.number_input("MU_ZONES", 1, 20, int(os.environ.get("MU_ZONES", "5")),
                                help="Emitting angle zones (default 5, use 1 for speed)")
    a_density = st.selectbox("A_DENSITY", [0, 1],
                              index=int(os.environ.get("A_DENSITY", "0")),
                              format_func=lambda x: {0: "0 — Constant density", 1: "1 — Shakura-Sunyaev zone A"}[x])
    rev_verb = st.selectbox("REV_VERB", [0, 1, 2],
                             index=int(os.environ.get("REV_VERB", "0")),
                             format_func=lambda x: {0: "0 — Silent", 1: "1 — Basic", 2: "2 — Verbose"}[x])

with st.sidebar.expander("Response & Reference Band"):
    rmf_path = st.text_input("RMF_SET", value=os.environ.get("RMF_SET", ""), help="Path to RMF file")
    arf_path = st.text_input("ARF_SET", value=os.environ.get("ARF_SET", ""), help="Path to ARF file")
    emin_ref = st.number_input("EMIN_REF (keV)", 0.01, 100.0,
                                float(os.environ.get("EMIN_REF", "0.3")), 0.1)
    emax_ref = st.number_input("EMAX_REF (keV)", 0.1, 1000.0,
                                float(os.environ.get("EMAX_REF", "10.0")), 0.5)

with st.sidebar.expander("Simulation"):
    seed_sim = st.number_input("SEED_SIM", value=int(os.environ.get("SEED_SIM", "-2851043")),
                                step=1, format="%d")
    backscl = st.number_input("BACKSCL", 0.1, 10.0, float(os.environ.get("BACKSCL", "1.0")), 0.1)

if st.sidebar.button("Apply All Env Variables", type="primary"):
    os.environ["RELTRANS_TABLES"] = tables_dir
    os.environ["ION_ZONES"] = str(ion_zones)
    os.environ["MU_ZONES"] = str(mu_zones)
    os.environ["A_DENSITY"] = str(a_density)
    os.environ["REV_VERB"] = str(rev_verb)
    os.environ["EMIN_REF"] = str(emin_ref)
    os.environ["EMAX_REF"] = str(emax_ref)
    if rmf_path: os.environ["RMF_SET"] = rmf_path
    if arf_path: os.environ["ARF_SET"] = arf_path
    os.environ["SEED_SIM"] = str(seed_sim)
    os.environ["BACKSCL"] = str(backscl)
    st.sidebar.success("All environment variables applied")

# Sidebar footer
st.sidebar.markdown("---")
lib_status = "[OK] Loaded" if MODELS_AVAILABLE else "Click 'Run Model' to load"
st.sidebar.info(f"**RELTRANS Dashboard v3.0**\nModel: `{model_name}`\nFortran: {lib_status}\nPhysics: {'[OK]' if PHYSICS_AVAILABLE else '[FAIL]'}")

# ============================================================================
#  MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Setup", "Analysis & Plots", "Advanced", "Data Export", "Documentation"
])

# ============================================================================
#  TAB 1 — MODEL SETUP
# ============================================================================
with tab1:
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### Geometry")
        if "Double LP" in model_type:
            h1 = st.number_input("Height 1 (Rg)", 1.5, 100.0, 10.0, 0.5, key="h1")
            h2 = st.number_input("Height 2 (Rg)", 1.5, 100.0, 15.0, 0.5, key="h2")
        else:
            h = st.number_input("Height h (Rg)", 1.5, 100.0, 6.0, 0.5, key="h")
        a_spin = st.slider("Spin (a)", -0.998, 0.998, 0.998, 0.001, help="Dimensionless spin parameter")
        inc = st.slider("Inclination (°)", 0.0, 85.0, 30.0, 1.0)
        rin = st.number_input("Rin (Rg)", -100.0, 100.0, -1.0, 0.1, help="Negative = ISCO")
        rout = st.number_input("Rout (Rg)", 10.0, 10000.0, 1000.0, 10.0)
        zcos = st.number_input("Redshift z", 0.0, 5.0, 0.0, 0.001)

    with col2:
        st.markdown("### Physics")
        Gamma = st.slider("Photon Index Γ", 1.0, 3.5, 2.0, 0.01)
        if "Distance" in model_type:
            Dkpc = st.number_input("Distance (kpc)", 0.1, 1000.0, 100.0, 1.0)
        else:
            logxi = st.slider("log(ξ)", 0.0, 5.0, 2.0, 0.1, help="Ionisation parameter")
        Afe = st.slider("Fe Abundance (A_Fe)", 0.5, 10.0, 1.0, 0.1)
        if "nthComp" in model_type or "Double" in model_type or "Distance" in model_type:
            lognep = st.slider("log(n_e) cm⁻³", 15.0, 19.0, 15.0, 0.1)
            kTe = st.slider("kTe (keV)", 10.0, 400.0, 60.0, 5.0, help="Corona electron temperature")
        elif "Power-law" in model_type:
            Ecut = st.slider("Ecut (keV)", 20.0, 500.0, 300.0, 10.0)
        if "Double LP" in model_type:
            eta_0 = st.slider("η₀", 0.0, 5.0, 0.5, 0.01)
            eta = st.slider("η(ν)", 0.0, 5.0, 1.0, 0.01)
            beta_p = st.slider("β_prop", 0.0, 1.0, 0.0, 0.01)

    with col3:
        st.markdown("### Observational")
        Nh = st.number_input("N_H (10²² cm⁻²)", 0.0, 100.0, 0.0, 0.1)
        boost = st.number_input("Boost (reflection norm)", 0.01, 10.0, 1.0, 0.1)
        Mass = st.number_input("Mass (M☉)", 1.0, 1e11, 1e7, 1e5, format="%.2e")
        st.markdown("### Timing")
        use_timing = st.checkbox("Enable timing analysis", value=False)
        if use_timing:
            floHz = st.number_input("f_lo (Hz)", 0.0, 1e4, 0.1, 0.01, format="%.4f")
            fhiHz = st.number_input("f_hi (Hz)", 0.0, 1e4, 1.0, 0.1, format="%.4f")
            ReIm = st.selectbox("Output product", [0,1,2,3,4,5,6],
                format_func=lambda x: ["Time-averaged","Real(C)","Imag(C)","|C|","Lag-energy",
                                        "|C| folded","Lag folded"][x])
            DelA = st.slider("φ_A", -np.pi, np.pi, 0.0, 0.01)
            DelAB = st.slider("φ_AB", -1.0, 1.0, 0.0, 0.01)
            g_var = st.slider("g (norm ratio)", 0.0, 1.0, 0.0, 0.01)
        else:
            floHz = fhiHz = 0.0; ReIm = 0; DelA = DelAB = g_var = 0.0
        resp = st.selectbox("Instrument Response", [0, 1], format_func=lambda x: {0:"None",1:"RMF/ARF"}[x])

    # --- Compute GR preview ---
    if PHYSICS_AVAILABLE:
        st.markdown("---")
        st.markdown("### GR Preview (live from parameters)")
        isco = KerrMetric.disco(a_spin)
        gso = KerrMetric.dgsofac(a_spin, h if "Double" not in model_type else h1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ISCO", f"{isco:.3f} Rg")
        c2.metric("g (src→obs)", f"{gso:.4f}")
        c3.metric("Spin", f"{a_spin:.3f}")
        c4.metric("Height", f"{h if 'Double' not in model_type else h1:.1f} Rg")

# ============================================================================
#  TAB 2 — ANALYSIS & PLOTS
# ============================================================================
with tab2:
    st.markdown("### Plot Selection")
    st.markdown("Select which plots to generate. Use **Combined** for a single multi-panel figure, or **Separate** for individual downloadable plots.")

    col_mode, col_opts = st.columns([1, 3])
    with col_mode:
        plot_mode = st.radio("Layout mode", ["Separate", "Combined"], help="Separate = individual plots, Combined = one multi-panel figure")

    with col_opts:
        # Callback to toggle all plot checkboxes
        def toggle_all_plots():
            val = st.session_state.chk_all_plots
            for k in ["chk_spectrum", "chk_reflection", "chk_continuum", "chk_gfactor", 
                      "chk_transfer", "chk_lag", "chk_emissivity", "chk_ionization", "chk_cumulative"]:
                st.session_state[k] = val

        st.checkbox("Select All Plots", key="chk_all_plots", on_change=toggle_all_plots)
        c1, c2, c3 = st.columns(3)
        with c1:
            chk_spectrum = st.checkbox("Energy Spectrum", key="chk_spectrum")
            chk_reflection = st.checkbox("XILLVER Reflection", key="chk_reflection")
            chk_continuum = st.checkbox("Power-Law Continuum", key="chk_continuum")
        with c2:
            chk_gfactor = st.checkbox("GR g-factor Profiles", key="chk_gfactor")
            chk_transfer = st.checkbox("Transfer Function Ψ(t)", key="chk_transfer")
            chk_lag = st.checkbox("Lag-Energy Spectrum", key="chk_lag")
        with c3:
            chk_emissivity = st.checkbox("Emissivity Profile", key="chk_emissivity")
            chk_ionization = st.checkbox("Ionisation Profile", key="chk_ionization")
            chk_cumulative = st.checkbox("Cumulative Flux", key="chk_cumulative")

    st.markdown("---")
    col_run1, col_run2 = st.columns([1, 1])

    with col_run1:
        st.markdown("#### Energy Grid")
        emin = st.number_input("E_min (keV)", 0.01, 50.0, 0.1, 0.01, key="emin")
        emax = st.number_input("E_max (keV)", 1.0, 1000.0, 100.0, 1.0, key="emax")
        nbins = st.slider("Energy bins", 50, 2000, 500, 10, key="nbins")

    with col_run2:
        st.markdown("#### Display Options")
        plot_type = st.selectbox("Spectrum format", ["E²F(E)", "F(E)", "νF(ν)", "Normalised"])
        log_scale = st.checkbox("Log scale", value=True, key="log2")

    # === RUN MODEL ===
    run_model = st.button("Run Model & Generate Plots", type="primary", width="stretch")

    # === PHYSICS-ONLY PLOTS (no Fortran needed) ===
    run_physics = st.button("Generate Physics-Only Plots (no Fortran)", width="stretch")

    if run_physics and PHYSICS_AVAILABLE:
        with st.spinner("Computing GR physics..."):
            phys_args = dict(a=a_spin, h=h if "Double" not in model_type else h1,
                             inc=inc, Gamma=Gamma, Ecut=Ecut if "Power-law" in model_type else 300.0,
                             logxi=logxi if "Distance" not in model_type else 2.0)
            phys_data = cached_precompute_all(tables_dir)
            # Override params with user values
            phys_data['params'].update(phys_args)
            # Recompute with user params
            r = np.linspace(KerrMetric.disco(a_spin) + 0.1, 50, 200)
            phys_data['r'] = r
            phys_data['gsd'] = KerrMetric.dglpfacthick(r, a_spin, phys_args['h'])
            phys_data['gdo'] = KerrMetric.dlgfacthick(a_spin, np.cos(np.radians(inc)), 0.0, r)
            phys_data['lorentz'] = KerrMetric.dlorfac(r, a_spin)
            phys_data['area'] = KerrMetric.dareafac(r, a_spin)
            phys_data['gso'] = KerrMetric.dgsofac(a_spin, phys_args['h'])
            phys_data['isco'] = KerrMetric.disco(a_spin)
            phys_data['E_cont'] = np.logspace(-0.5, 2.5, 500)
            phys_data['continuum'] = powerlaw_continuum(phys_data['E_cont'], Gamma, phys_args['Ecut'])
            tf = TransferFunction(a=a_spin, h=phys_args['h'], inc=inc)
            t_grid, psi = tf.impulse_response(nt=400)
            phys_data['t_tf'] = t_grid; phys_data['psi_tf'] = psi
            E_lag = np.logspace(-0.3, 1.5, 150)
            phys_data['E_lag'] = E_lag
            phys_data['lag'] = tf.lag_energy(E_lag, Gamma=Gamma)
            st.session_state.phys_data = phys_data

        # Generate selected physics plots
        plots = {}
        pd_data = st.session_state.phys_data
        if chk_reflection: plots["XILLVER Reflection"] = plot_reflection_spectrum(pd_data)
        if chk_continuum: plots["Power-Law Continuum"] = plot_continuum(pd_data)
        if chk_gfactor: plots["GR g-factor Profiles"] = plot_gfactor_profiles(pd_data)
        if chk_transfer: plots["Transfer Function"] = plot_transfer_function(pd_data)
        if chk_lag: plots["Lag-Energy Spectrum"] = plot_lag_energy(pd_data)
        if chk_emissivity: plots["Emissivity Profile"] = plot_emissivity(pd_data)
        if chk_ionization: plots["Ionisation Profile"] = plot_ionization_profile(pd_data)

        if plots:
            st.session_state.plot_history.append({
                'title': "Physics-only",
                'plots': plots,
                'plot_mode': plot_mode
            })
            st.session_state.history_idx = len(st.session_state.plot_history) - 1
            st.success("Physics plots generated successfully")
        else:
            st.info("Select at least one plot checkbox above")

    if run_model:
        if not load_models():
            st.error("Fortran library not loaded. Ensure `make` has been run and compiled object exists.")
            st.info("You can still use **Physics-Only Plots** above without the Fortran library.")
        else:
            model_func = globals().get(model_func_name)
            if model_func is None:
                st.error(f"Model function `{model_func_name}` not found")
            else:
                with st.spinner(f"Running {model_name}..."):
                    try:
                        # CRITICAL: Export environment variables required by Fortran XSPEC/Timing models
                        # Otherwise Fortran will pause and wait for interactive stdin, hanging the dashboard
                        os.environ["RELTRANS_TABLES"] = tables_dir
                        os.environ["EMIN_REF"] = str(emin_ref)
                        os.environ["EMAX_REF"] = str(emax_ref)
                        
                        ear = np.logspace(np.log10(emin), np.log10(emax), nbins+1).astype(np.float32)
                        # Build parameter array
                        if "Double LP" in model_type:
                            params = np.array([h1, h2, a_spin, inc, rin, rout, zcos, Gamma,
                                logxi, Afe, lognep, kTe, eta_0, eta, beta_p, Nh, boost, Mass,
                                floHz, fhiHz, ReIm, DelA, 0.0, 0.0, 0.0, 0.0, resp], dtype=np.float32)
                        elif "Distance" in model_type:
                            params = np.array([h, a_spin, inc, rin, rout, zcos, Gamma, Dkpc, Afe,
                                lognep, kTe, Nh, 1.0, Mass, 0.0, 0.0, 0.0, floHz, fhiHz, ReIm,
                                DelA, DelAB, g_var, 1.0, resp], dtype=np.float32)
                        elif "Power-law" in model_type:
                            params = np.array([h, a_spin, inc, rin, rout, zcos, Gamma, logxi, Afe,
                                Ecut, Nh, boost, Mass, floHz, fhiHz, ReIm, DelA, DelAB, g_var, resp],
                                dtype=np.float32)
                        else:  # DCp
                            params = np.array([h, a_spin, inc, rin, rout, zcos, Gamma, logxi, Afe,
                                lognep, kTe, Nh, boost, Mass, floHz, fhiHz, ReIm, DelA, DelAB, g_var, resp],
                                dtype=np.float32)

                        print(f"DEBUG: Calling {model_func_name} with params={params}...")
                        photar = model_func(ear, params)
                        print(f"DEBUG: Fortran returned {len(photar)} values.")
                        
                        E = 0.5 * (ear[1:] + ear[:-1])
                        dE = ear[1:] - ear[:-1]
                        flux = photar / dE
                        st.session_state.results = {'E': E, 'flux': flux, 'photar': photar,
                                                      'ear': ear, 'params': params, 'model': model_name}

                        # Also compute physics data
                        if PHYSICS_AVAILABLE:
                            print("DEBUG: Starting precompute_all...")
                            phys_args = dict(a=a_spin, h=h if "Double" not in model_type else h1,
                                             inc=inc, Gamma=Gamma, Ecut=Ecut if "Power-law" in model_type else 300.0,
                                             logxi=logxi if "Distance" not in model_type else 2.0)
                            phys_data = cached_precompute_all(tables_dir)
                            phys_data['params'].update(phys_args)
                            
                            print("DEBUG: Computing physics grids...")
                            r = np.linspace(KerrMetric.disco(a_spin) + 0.1, 50, 200)
                            phys_data['r'] = r
                            phys_data['gsd'] = KerrMetric.dglpfacthick(r, a_spin, phys_args['h'])
                            phys_data['gdo'] = KerrMetric.dlgfacthick(a_spin, np.cos(np.radians(inc)), 0.0, r)
                            phys_data['lorentz'] = KerrMetric.dlorfac(r, a_spin)
                            phys_data['area'] = KerrMetric.dareafac(r, a_spin)
                            phys_data['gso'] = KerrMetric.dgsofac(a_spin, phys_args['h'])
                            phys_data['isco'] = KerrMetric.disco(a_spin)
                            phys_data['E_cont'] = np.logspace(-0.5, 2.5, 500)
                            phys_data['continuum'] = powerlaw_continuum(phys_data['E_cont'], Gamma, phys_args['Ecut'])
                            
                            print("DEBUG: Computing TransferFunction...")
                            tf = TransferFunction(a=a_spin, h=phys_args['h'], inc=inc)
                            t_grid, psi = tf.impulse_response(nt=400)
                            phys_data['t_tf'] = t_grid; phys_data['psi_tf'] = psi
                            
                            print("DEBUG: Computing lag_energy...")
                            E_lag = np.logspace(-0.3, 1.5, 150)
                            phys_data['E_lag'] = E_lag
                            phys_data['lag'] = tf.lag_energy(E_lag, Gamma=Gamma)
                            
                            st.session_state.phys_data = phys_data

                        # Generate plots
                        print("DEBUG: Starting chart generation...")
                        plots = {}
                        if chk_spectrum:
                            print("DEBUG: Plotting Energy Spectrum...")
                            plots["Energy Spectrum"] = plot_energy_spectrum(E, flux, model_name, plot_type, log_scale)
                        if chk_reflection and PHYSICS_AVAILABLE:
                            print("DEBUG: Plotting XILLVER...")
                            plots["XILLVER Reflection"] = plot_reflection_spectrum(st.session_state.phys_data)
                        if chk_continuum and PHYSICS_AVAILABLE:
                            print("DEBUG: Plotting Continuum...")
                            plots["Power-Law Continuum"] = plot_continuum(st.session_state.phys_data)
                        if chk_gfactor and PHYSICS_AVAILABLE:
                            print("DEBUG: Plotting g-factors...")
                            plots["GR g-factor Profiles"] = plot_gfactor_profiles(st.session_state.phys_data)
                        if chk_transfer and PHYSICS_AVAILABLE:
                            print("DEBUG: Plotting Transfer Function...")
                            plots["Transfer Function"] = plot_transfer_function(st.session_state.phys_data)
                        if chk_lag and PHYSICS_AVAILABLE:
                            print("DEBUG: Plotting Lag...")
                            plots["Lag-Energy Spectrum"] = plot_lag_energy(st.session_state.phys_data)
                        if chk_emissivity and PHYSICS_AVAILABLE:
                            print("DEBUG: Plotting Emissivity...")
                            plots["Emissivity Profile"] = plot_emissivity(st.session_state.phys_data)
                        if chk_ionization and PHYSICS_AVAILABLE:
                            print("DEBUG: Plotting Ionization...")
                            plots["Ionisation Profile"] = plot_ionization_profile(st.session_state.phys_data)
                        if chk_cumulative:
                            print("DEBUG: Plotting Cumulative Flux...")
                            plots["Cumulative Flux"] = plot_cumulative_flux(E, photar, log_scale)

                        print("DEBUG: All plots generated successfully!")
                        # Metrics row
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Total Flux", f"{np.sum(photar):.3e}")
                        m2.metric("Peak Energy", f"{E[np.argmax(flux)]:.2f} keV")
                        m3.metric("Peak Flux", f"{np.max(flux):.3e}")
                        m4.metric("Mean Energy", f"{np.average(E, weights=np.abs(flux)+1e-30):.2f} keV")
                        m5.metric("Bins", len(E))
                        
                        if plots:
                            st.session_state.plot_history.append({
                                'title': f"{model_name} (Γ={Gamma}, h={h if 'Double' not in model_type else h1}, a={a_spin})",
                                'plots': plots,
                                'plot_mode': plot_mode
                            })
                            st.session_state.history_idx = len(st.session_state.plot_history) - 1
                            st.success(f"[OK] {model_name} computed successfully — {len(plots)} plots generated")
                        else:
                            st.info("Select at least one plot checkbox above")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # === DISPLAY PLOT HISTORY ===
    if len(st.session_state.plot_history) > 0:
        st.markdown("---")
        st.markdown("### Plot History Book")
        
        hist_col1, hist_col2, hist_col3 = st.columns([1, 2, 1])
        with hist_col1:
            if st.button("⬅Previous Plot", disabled=st.session_state.history_idx <= 0, width="stretch"):
                st.session_state.history_idx -= 1
        with hist_col2:
            st.markdown(f"<div style='text-align: center; color: #8b949e;'>Showing page <b>{st.session_state.history_idx + 1}</b> of <b>{len(st.session_state.plot_history)}</b></div>", unsafe_allow_html=True)
            active_data = st.session_state.plot_history[st.session_state.history_idx]
            st.markdown(f"<div style='text-align: center; color: #58a6ff; font-weight: bold;'>{active_data['title']}</div>", unsafe_allow_html=True)
        with hist_col3:
            if st.button("Next Plot", disabled=st.session_state.history_idx >= len(st.session_state.plot_history) - 1, width="stretch"):
                st.session_state.history_idx += 1
                
        # Clamp memory pointers to prevent IndexErrors during rapid reruns
        st.session_state.history_idx = max(0, min(st.session_state.history_idx, len(st.session_state.plot_history) - 1))
        
        # Render the active historically-saved plots
        active_hist = st.session_state.plot_history[st.session_state.history_idx]
        active_plots = active_hist['plots']
        if active_hist['plot_mode'] == "Combined":
            combined_fig, combined_png = make_combined_figure(active_plots)
            if combined_fig:
                st.pyplot(combined_fig)
                st.download_button("Download Combined Figure (PNG)", combined_png,
                                    "reltrans_combined.png", "image/png", key=f"dl_combined_hist_{st.session_state.history_idx}")
        else:
            for i, (name, (fig, csv_data)) in enumerate(active_plots.items()):
                st.pyplot(fig)
                make_download_row(fig, csv_data, name.replace(" ", "_").lower())
                st.markdown("---")

# ============================================================================
#  TAB 3 — ADVANCED
# ============================================================================
with tab3:
    st.markdown("### Advanced Analysis")
    if st.session_state.results:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Parameter Scan")
            scan_param = st.selectbox("Scan parameter", ["Spin (a)","Inclination","Height (h)","Γ","log(ξ)"])
            sc1, sc2 = st.columns(2)
            scan_min = sc1.number_input("Min", value=0.0, key="sc_min")
            scan_max = sc2.number_input("Max", value=0.998, key="sc_max")
            scan_steps = st.slider("Steps", 3, 50, 10, key="sc_steps")
            if st.button("Run Parameter Scan", type="primary"):
                if not load_models():
                    st.error("Fortran library required for parameter scan")
                else:
                    with st.spinner(f"Scanning {scan_param}..."):
                        model_func = globals()[model_func_name]
                        params = st.session_state.results['params'].copy()
                        ear = st.session_state.results['ear']
                        pidx = {"Spin (a)":1 if "Double" not in model_type else 2,
                                "Inclination":2 if "Double" not in model_type else 3,
                                "Height (h)":0, "Γ":6 if "Double" not in model_type else 7,
                                "log(ξ)":7 if "Double" not in model_type else 8}[scan_param]
                        scan_vals = np.linspace(scan_min, scan_max, scan_steps)
                        apply_plot_style()
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        E = 0.5 * (ear[1:] + ear[:-1])
                        total_fluxes = []
                        for i, val in enumerate(scan_vals):
                            p = params.copy(); p[pidx] = val
                            photar = model_func(ear, p)
                            flux = photar / (ear[1:] - ear[:-1])
                            col = plt.cm.plasma(i / len(scan_vals))
                            ax1.loglog(E, flux * E**2, color=col, lw=1.5, alpha=0.8)
                            total_fluxes.append(np.sum(photar))
                        ax1.set_xlabel("Energy (keV)", fontweight='bold')
                        ax1.set_ylabel("E² × F(E)", fontweight='bold')
                        ax1.set_title(f"{scan_param} scan", fontweight='bold')
                        ax1.grid(True, alpha=0.3)
                        ax2.plot(scan_vals, total_fluxes, 'o-', color=COLORS[0], lw=2, ms=6)
                        ax2.set_xlabel(scan_param, fontweight='bold')
                        ax2.set_ylabel("Total Flux", fontweight='bold')
                        ax2.set_title("Flux vs Parameter", fontweight='bold')
                        ax2.grid(True, alpha=0.3)
                        fig.tight_layout()
                        st.pyplot(fig)
                        make_download_row(fig, pd.DataFrame({"param": scan_vals, "total_flux": total_fluxes}).to_csv(index=False), "param_scan")
                        st.success(f"Scan complete: {scan_steps} models")

        with col2:
            st.markdown("#### Spectral Fitting (CSV upload)")
            uploaded = st.file_uploader("Upload spectrum (CSV: Energy, Flux, Error)", type=['csv'])
            if uploaded:
                obs = pd.read_csv(uploaded)
                st.dataframe(obs.head(), width="stretch")
            st.markdown("#### XSPEC Export")
            st.code(f"lmod xsreltrans ./build/xspec\n# Model: {model_name}", language="bash")
    else:
        st.info("Run a model first (Tab 2) to enable advanced analysis.")

# ============================================================================
#  TAB 4 — DATA EXPORT
# ============================================================================
with tab4:
    st.markdown("### Data Export")
    if st.session_state.results:
        res = st.session_state.results
        df = pd.DataFrame({
            'Energy_keV': res['E'], 'Flux': res['flux'],
            'E2FE': res['flux'] * res['E']**2, 'Photar': res['photar']
        })
        st.dataframe(df, width="stretch", height=400)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download CSV", df.to_csv(index=False),
                                "reltrans_spectrum.csv", "text/csv", key="exp_csv")
        with c2:
            # FITS export
            try:
                from astropy.io import fits as pyfits
                col1f = pyfits.Column(name='ENERGY', format='E', array=res['E'])
                col2f = pyfits.Column(name='FLUX', format='E', array=res['flux'])
                col3f = pyfits.Column(name='PHOTAR', format='E', array=res['photar'])
                hdu = pyfits.BinTableHDU.from_columns([col1f, col2f, col3f])
                hdu.header['MODEL'] = model_name
                hdu.header['SPIN'] = (float(res['params'][1 if "Double" not in model_type else 2]), 'BH spin')
                buf = io.BytesIO(); hdu.writeto(buf, overwrite=True); buf.seek(0)
                st.download_button("Download FITS", buf.getvalue(),
                                    "reltrans_spectrum.fits", "application/fits", key="exp_fits")
            except ImportError:
                st.warning("Install astropy for FITS export: `pip install astropy`")
        with c3:
            param_lines = [f"# Model: {model_name}"]
            for i, v in enumerate(res['params']):
                param_lines.append(f"param[{i}] = {v}")
            st.download_button("Download Parameters", "\n".join(param_lines),
                                "reltrans_params.txt", "text/plain", key="exp_params")
    else:
        st.info("No data available. Run a model in the Analysis tab first.")

# ============================================================================
#  TAB 5 — DOCUMENTATION
# ============================================================================
with tab5:
    st.markdown("""
    ### RELTRANS Model Flavours

    | Model | Continuum | Density | Tables |
    |-------|-----------|---------|--------|
    | `reltransDCp` | nthComp | Free param | xillverCp, xillverD-5 |
    | `reltransPL` | Power-law+cutoff | Fixed 10¹⁵ | xillver-a-Ec5 |
    | `reltransx` | nthComp | Free | ReflionX tables |
    | `reltransDbl` | nthComp (2 sources) | Free | xillverCp, xillverD-5 |
    | `rtdist` | nthComp | Free | xillverCp, xillverD-5 |

    ### Timing Products (ReIm parameter)
    | Value | Product |
    |-------|---------|
    | 0 | Time-averaged spectrum |
    | 1 | Real part of cross-spectrum |
    | 2 | Imaginary part |
    | 3 | Modulus |
    | 4 | Lag-energy spectrum |
    | 5 | Modulus (folded) |
    | 6 | Lag-energy (folded) |

    ### Environment Variables
    ```bash
    export RELTRANS_TABLES="/path/to/tables"
    export ION_ZONES=20        # radial ionisation zones
    export MU_ZONES=1          # angle zones (1 for speed)
    export A_DENSITY=0         # 0=constant, 1=Shakura-Sunyaev
    export REV_VERB=0          # verbosity
    export EMIN_REF=0.3        # reference band min (keV)
    export EMAX_REF=10.0       # reference band max (keV)
    export RMF_SET=/path.rmf   # response matrix
    export ARF_SET=/path.arf   # effective area
    ```

    ### Key Functions
    - **GR**: `disco()`, `dgsofac()`, `dglpfacthick()`, `dlgfacthick()`, `dlorfac()`, `dareafac()`
    - **Reflection**: `rest_frame()` → `get_xillver()`, `radfunctions_dens()`
    - **Convolution**: `do_convolutions()` → `conv_one_FFTw()`, `rawS()`, `rawG()`
    - **Output**: `propercross()`, `crebin()`, `cfoldandbin()`

    ### Citations
    - Ingram et al. (2019) MNRAS 488, 324–347
    - Mastroserio et al. (2021) MNRAS 507, 55–73
    - Mastroserio et al. (2022) MNRAS 514, 2813–2828
    - Lucchini et al. (2023) ApJ 951, 19

    ### Dashboard Info
    - Built with Streamlit + Matplotlib
    - Physics module: `reltrans_physics.py` (GR formulas ported from `GR_factors.f90`)
    - XILLVER data: reads normalised FITS tables directly via `astropy.io.fits`
    """)
