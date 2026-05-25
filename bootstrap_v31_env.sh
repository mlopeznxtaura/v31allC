#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] root=$ROOT_DIR"

echo "[bootstrap] checking python"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[bootstrap] error: python not found" >&2
  exit 1
fi
$PY --version

echo "[bootstrap] creating .venv"
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel >/dev/null

if [ -f requirements.txt ]; then
  echo "[bootstrap] installing requirements"
  python -m pip install -r requirements.txt >/dev/null || true
fi
python -m pip install pytest >/dev/null || true

echo "[bootstrap] network checks"
python - << 'PY'
import socket, ssl, urllib.request
hosts=["github.com","pypi.org","files.pythonhosted.org"]
for h in hosts:
    try:
        socket.gethostbyname(h)
        print(f"[net] dns ok: {h}")
    except Exception as e:
        print(f"[net] dns fail: {h} ({e})")
for u in ["https://github.com","https://pypi.org/simple"]:
    try:
        with urllib.request.urlopen(u, timeout=8) as r:
            print(f"[net] http ok: {u} -> {r.status}")
    except Exception as e:
        print(f"[net] http fail: {u} ({e})")
PY

echo "[bootstrap] done. activate with: source .venv/bin/activate"
