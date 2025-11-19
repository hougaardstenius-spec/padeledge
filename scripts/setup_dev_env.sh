#!/usr/bin/env bash

set -e

echo "==============================="
echo "🔥 PadelEdge Dev Environment Setup"
echo "==============================="

# --------------------------------------------
# 1. Ensure Homebrew exists
# --------------------------------------------
if ! command -v brew &> /dev/null; then
    echo "⚠️ Homebrew ikke fundet — installerer..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✔ Homebrew OK"
fi

# --------------------------------------------
# 2. Install Python 3.10 via brew
# --------------------------------------------
if ! brew ls --versions python@3.10 >/dev/null; then
    echo "📦 Installerer Python 3.10..."
    brew install python@3.10
else
    echo "✔ Python@3.10 findes allerede"
fi

PYTHON_BIN="/opt/homebrew/bin/python3.10"

echo "➡ Bruger Python binary: $PYTHON_BIN"

# --------------------------------------------
# 3. Remove old venv if any
# --------------------------------------------
echo "🧹 Rydder gammel .venv..."
rm -rf .venv

# --------------------------------------------
# 4. Create venv with Python 3.10
# --------------------------------------------
echo "🐍 Opretter nyt .venv med Python 3.10..."
$PYTHON_BIN -m venv .venv

# --------------------------------------------
# 5. Activate venv
# --------------------------------------------
echo "🔌 Aktiverer venv..."
source .venv/bin/activate

echo "✔ Python version i venv:"
python --version

# --------------------------------------------
# 6. Install pip deps
# --------------------------------------------
echo "📦 Installerer pip dependencies..."
pip install --upgrade pip wheel setuptools

pip install -r requirements.txt || {
    echo "❌ Requirements failed — MediaPipe muligvis ikke kompatibel"
}

# --------------------------------------------
# 7. Test MediaPipe
# --------------------------------------------
echo "🔍 Tester MediaPipe install..."
python - << 'EOF'
import sys
try:
    import mediapipe as mp
    print("✔ MediaPipe import OK")
except Exception as e:
    print("❌ MediaPipe fejler:", e)
    print("Python version:", sys.version)
EOF

echo "==============================="
echo "🎉 Setup færdigt!"
echo "==============================="
echo "👉 Husk at aktivere miljøet:"
echo "source .venv/bin/activate"
