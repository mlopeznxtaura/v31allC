#!/usr/bin/env bash
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

export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
python3 /mnt/d/NextAura/v31all_1/v31allC/inspect_pt.py
