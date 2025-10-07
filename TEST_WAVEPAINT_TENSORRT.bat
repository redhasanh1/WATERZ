@echo off
cd /d "%~dp0"

echo ================================================================
echo Test WavePaint TensorRT on Video
echo ================================================================
echo.
echo Expected speed: 0.5 seconds/frame (10-20x faster!)
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

python -u test_wavepaint_tensorrt.py

pause
