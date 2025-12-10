@echo off
cd /d "%~dp0"

set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%PATH%

REM Use Python 3.12 with CUDA torch
set PYTHON_CMD=py -3.12

echo ======================================
echo Export YOLO v2 to TensorRT FP16
echo ======================================
echo.

%PYTHON_CMD% export_sora_v2_tensorrt.py

pause
