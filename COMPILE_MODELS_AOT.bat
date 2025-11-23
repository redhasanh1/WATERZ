@echo off
echo ========================================
echo AOT Model Compilation
echo ========================================
echo.
echo This compiles RAFT and RFCNet models ONCE
echo so 4 parallel workers can load them instantly
echo (no per-worker compilation delay)
echo.
echo This will take 1-2 minutes...
echo.

cd /d "%~dp0"

"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" compile_models_aot.py

echo.
pause
