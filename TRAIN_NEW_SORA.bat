@echo off
cd /d "%~dp0"

set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%PATH%

echo ======================================
echo Train YOLO on NEW Sora Watermarks
echo ======================================
echo.

python train_new_sora.py

pause
