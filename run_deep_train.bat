@echo off
cd /d D:\NextAura\v31all_1\v31allC
call C:\Users\archi\miniforge3\Scripts\activate.bat nextaura
python deep_train.py > deep_train_run.log 2>&1
echo EXIT:%ERRORLEVEL% >> deep_train_run.log
