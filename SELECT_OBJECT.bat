@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Add TensorRT DLLs to PATH for future TensorRT integration
set PATH=D:\watermarkz\TensorRT-10.13.3.9\bin;%PATH%

REM Use the correct Python 3.12 with all packages installed
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

echo ============================================================
echo SAM2 Fast Object Selection
echo ============================================================
echo.

"%PYTHON_PATH%" SELECT_OBJECT_FAST.py

echo.
echo ============================================================
echo.

pause
