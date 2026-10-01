@echo off
cd /d D:\NextAura\v31all_1\v31allC
call C:\Users\archi\miniforge3\Scripts\activate.bat nextaura
python -u full_cycle_v2.py > D:\NextAura\v31all_1\v31allC\full_cycle_v2.log 2>&1
echo DONE:%ERRORLEVEL% >> D:\NextAura\v31all_1\v31allC\full_cycle_v2.log
