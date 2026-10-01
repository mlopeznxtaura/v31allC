#!/usr/bin/env bash
cd /mnt/d/NextAura/v31all_1/v31allC
NVIDIA_LIBS=""
for dir in /usr/local/lib/python3.12/dist-packages/nvidia/*/lib; do [ -d "$dir" ] && NVIDIA_LIBS="$dir:$NVIDIA_LIBS"; done
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/d/NextAura/v31all_1/v31allC:${PYTHONPATH:-}"
python3 cycle_and_grow.py 2>&1
