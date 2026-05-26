#!/usr/bin/env bash
# run_gui.sh - Launch NiceGUI dashboard in WSL with correct CUDA library paths

# Resolve all local nvidia wheel library paths
NVIDIA_LIBS=""
for dir in /usr/local/lib/python3.12/dist-packages/nvidia/*/lib; do
    if [ -d "$dir" ]; then
        if [ -z "$NVIDIA_LIBS" ]; then
            NVIDIA_LIBS="$dir"
        else
            NVIDIA_LIBS="$NVIDIA_LIBS:$dir"
        fi
    fi
done

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"

echo "[INFO] Derived LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "[INFO] Launching NiceGUI on port 8080..."

# Insert python path to ensure we can load stage1/stage2 modules
export PYTHONPATH="/mnt/d/NextAura/v31all_1/v31allC:${PYTHONPATH:-}"

# Launch the app
python3 /mnt/d/NextAura/v31all_1/v31allC/gui/main.py
