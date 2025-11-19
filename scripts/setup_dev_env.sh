#!/usr/bin/env bash

echo "==============================="
echo "🔥 PadelEdge Auto Dev Setup"
echo "==============================="

# --------- CHECK BREW ---------
if ! command -v brew &> /dev/null
then
    echo "⚠️ Homebrew ikke fundet. Installerer..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✔ Homebrew OK"
fi

# --------- INSTALL PYENV ---------
if ! command -v pyenv &> /dev/null
then
    echo "⚠️ pyenv ikke fundet. Installerer..."
    brew install pyenv
else
    echo "✔ pyenv OK"
fi

# --------- INSTALL PYTHON 3.10 ---------
PYTHON_VERSION="3.10.13"

if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "📦 Installerer Python $PYTHON_VERSION..."
    CFLAGS="-I$(brew --prefix)/opt/zlib/include" \
    LDFLAGS="-L$(brew --prefix)/opt/zlib/lib" \
    pyenv install $PYTHON_VERSION
else
    echo "✔ Python $PYTHON_VERSION findes allerede"
fi

# --------- SET LOCALLY ---------
echo "📌 Sætter Python version lokalt..."
pyenv local $PYTHON_VERSION

# --------- CREATE VENV ---------
echo "🐍 Opretter .venv..."
rm -rf .venv
python -m venv .venv

echo "📌 Aktivér venv manuelt efter setup:"
echo "   source .venv/bin/activate"

# ------- INSTALL DEPENDENCIES -------
echo "📦 Installerer requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install --upgrade wheel setuptools

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ Ingen requirements.txt fundet!"
fi

# --------- CREATE .vscode SETTINGS ---------
echo "🛠 Opretter .vscode setup..."

mkdir -p .vscode

cat > .vscode/settings.json <<EOF
{
    "python.defaultInterpreterPath": "\${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "python.analysis.extraPaths": [
        "\${workspaceFolder}",
        "\${workspaceFolder}/utils",
        "\${workspaceFolder}/scripts"
    ]
}
EOF

cat > .vscode/extensions.json <<EOF
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.black-formatter",
        "ms-python.flake8",
        "ms-azuretools.vscode-docker"
    ]
}
EOF

# --------- MEDIA PIPE CHECK ---------
echo "🔍 Tester MediaPipe installation..."
python - << 'EOF'
try:
    import mediapipe as mp
    print("✔ MediaPipe OK")
except Exception as e:
    print("❌ MediaPipe kunne ikke importeres:", e)
EOF

# --------- FINAL ---------
echo "==============================="
echo "🎉 Dev Setup Færdigt!"
echo "==============================="
echo ""
echo "👉 Kør nu:"
echo "source .venv/bin/activate"
echo ""
echo "👉 VS Code bruger nu automatisk .venv"
echo ""
