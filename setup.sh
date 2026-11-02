#!/bin/bash
set -e
echo "========================================================"
echo "Setting up RetailIQ Enterprise Platform on Linux / Cloud"
echo "========================================================"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Launching Streamlit production server..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
