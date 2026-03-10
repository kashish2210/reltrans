#!/bin/bash
# Final Dashboard Launcher

export RELTRANS_TABLES="$(pwd)/tables"
export LD_LIBRARY_PATH="$(pwd)/build/lib:$LD_LIBRARY_PATH"
export ION_ZONES=20
export MU_ZONES=1
export REV_VERB=0

echo "🚀 Launching Reltrans Dashboard"
echo "📍 http://localhost:8501"
echo ""

~/.local/bin/streamlit run reltrans_pro_dashboard.py
