#!/bin/bash

# --- FINAL PRODUCTION BOOTSTRAPPER ---
# We use simple text to avoid Linux syntax errors
PROJECT_DIR="$HOME/Desktop/SignLanguageEdge"
VENV_PATH="$PROJECT_DIR/venv"

echo "=================================================="
echo "   AUTONOMOUS EDGE AI: SYSTEM STARTUP             "
echo "=================================================="

# 1. Check for Virtual Environment
if [ -d "$VENV_PATH" ]; then
    echo "STEP 1: Environment detected"
    source "$VENV_PATH/bin/activate"
else
    echo "STEP 1: Environment missing - creating now"
    cd "$PROJECT_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install opencv-python-headless streamlit psutil requests ai-edge-litert pandas matplotlib
fi

# 2. Bluetooth Connection
echo "STEP 2: Connecting to Bluetooth Speaker"
# Replace MAC with your speaker: A2:75:E5:3B:61:2B
bluetoothctl connect A2:75:E5:3B:61:2B || echo "Warning: Bluetooth connection failed"

# 3. Launch Dashboard
echo "=================================================="
echo "   LAUNCHING DASHBOARD                            "
echo "=================================================="

cd "$PROJECT_DIR"
python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0