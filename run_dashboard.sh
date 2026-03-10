#!/bin/bash
# Wrapper to run dashboard with stdin redirected

# Create a named pipe for stdin
mkfifo /tmp/dashboard_stdin 2>/dev/null || true

# Feed empty responses to the pipe in background
(while true; do echo ""; sleep 1; done) > /tmp/dashboard_stdin &
FEEDER_PID=$!

# Run streamlit with redirected stdin
streamlit run reltrans_pro_dashboard.py --server.address=0.0.0.0 < /tmp/dashboard_stdin

# Cleanup
kill $FEEDER_PID 2>/dev/null
rm /tmp/dashboard_stdin 2>/dev/null
