#!/bin/bash
export HOME=/home/archi
source /home/archi/miniforge3/etc/profile.d/conda.sh
conda activate nextaura
cd /mnt/d/NextAura/v31all_1/v31allC

LOG=/mnt/d/NextAura/v31all_1/v31allC/pipeline.log
echo "=== PIPELINE START $(date) ===" > $LOG

echo "--- STEP 1: cycle_and_grow ---" >> $LOG
python -u cycle_and_grow.py >> $LOG 2>&1
echo "cycle_and_grow EXIT:$?" >> $LOG

echo "--- STEP 2: grow_corpus_web ---" >> $LOG
python -u grow_corpus_web.py >> $LOG 2>&1
echo "grow_corpus_web EXIT:$?" >> $LOG

echo "--- STEP 3: inject mp4 frames ---" >> $LOG
python -u patch_frames.py >> $LOG 2>&1
echo "patch_frames EXIT:$?" >> $LOG

echo "--- STEP 4: deep_train ---" >> $LOG
python -u deep_train.py >> $LOG 2>&1
echo "deep_train EXIT:$?" >> $LOG

echo "=== PIPELINE DONE $(date) ===" >> $LOG
