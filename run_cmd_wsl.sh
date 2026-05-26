#!/usr/bin/env bash
# run_cmd_wsl.sh - Run any NextAura script with correct CUDA and Python environment

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

export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/d/NextAura/v31all_1/v31allC:${PYTHONPATH:-}"

python3 "$@"
