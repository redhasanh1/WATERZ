@echo off
call SETUP_TRT_ENV.bat >nul 2>&1
%LOCALAPPDATA%\Programs\Python\Python312\python.exe test_v3_quality.py
pause
