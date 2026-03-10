#!/bin/bash
# Run dashboard in Docker with HEASOFT

docker run -it --rm \
  -v /mnt/k/Gsoc2026/reltrans:/workspace \
  -v reltrans-venv:/venv \
  -p 8501:8501 \
  -w /workspace \
  fjebaker/heasoft:ci-6.33.1-amd64 \
  bash -c "
    source /opt/heasoft/headas-init.sh && \
    if [ ! -f /venv/bin/activate ]; then \
      echo 'Creating virtual environment...' && \
      apt-get update -qq && apt-get install -y -qq python3-venv && \
      python3 -m venv /venv && \
      . /venv/bin/activate && \
      pip install --upgrade pip && \
      pip install streamlit numpy matplotlib pandas scipy; \
    else \
      echo 'Using existing virtual environment...'; \
    fi && \
    . /venv/bin/activate && \
    export RELTRANS_TABLES=/workspace/tables && \
    export ION_ZONES=20 && \
    export MU_ZONES=1 && \
    export REV_VERB=0 && \
    chmod +x /workspace/run_dashboard.sh && \
    /workspace/run_dashboard.sh
  "
