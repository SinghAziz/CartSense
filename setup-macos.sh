#!/bin/sh
set -eu

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        case "$version" in 3.10|3.11|3.12) PYTHON=$(command -v "$candidate"); break ;; esac
    fi
done

if [ -z "$PYTHON" ]; then
    echo "CartSense requires Python 3.10–3.12. Install Python 3.12 and rerun this script." >&2
    exit 1
fi

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "Setup complete. Activate with: source .venv/bin/activate"
