#!/bin/bash
export HOME=/home/archi
source /home/archi/miniforge3/etc/profile.d/conda.sh
conda activate nextaura
cd /mnt/d/NextAura/v31all_1/v31allC
python -u full_cycle_v2.py > /mnt/d/NextAura/v31all_1/v31allC/full_cycle_v2.log 2>&1
echo "EXIT:$?" >> /mnt/d/NextAura/v31all_1/v31allC/full_cycle_v2.log
