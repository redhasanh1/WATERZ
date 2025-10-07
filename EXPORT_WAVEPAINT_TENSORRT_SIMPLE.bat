@echo off
cd /d "%~dp0"

echo ================================================================
echo Export WavePaint to TensorRT - Simple Version
echo ================================================================
echo.
echo This creates the TensorRT engine for 10-20x speedup
echo Takes 2-5 minutes...
echo.

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache
set PYTHONPATH=%~dp0python_packages
set PYTHONUNBUFFERED=1

REM Add TensorRT DLLs to PATH
set PATH=%~dp0python_packages;%PATH%

python -u export_wavepaint_tensorrt_simple.py

pause
