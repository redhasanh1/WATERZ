@echo off
echo ========================================
echo Installing torch-tensorrt for AOT Compilation
echo ========================================
echo.
echo This enables pre-compiling models for instant worker loading
echo (0.5s instead of 30-60s per worker)
echo.

cd /d "%~dp0"

REM Use the correct Python
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip install torch-tensorrt --extra-index-url https://pypi.ngc.nvidia.com

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: python build_compiled_models.py
echo 2. Wait for compilation (60-120s one-time)
echo 3. Start workers - they load in 0.5s!
echo.

pause
