#!/usr/bin/env bash
cd /mnt/d/NextAura/v31all_1/v31allC
NVIDIA_LIBS=
export LD_LIBRARY_PATH=
export PYTHONPATH=/mnt/d/NextAura/v31all_1/v31allC
python3 cycle_and_grow.py 2>&1
