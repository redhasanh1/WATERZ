@echo off
cd /d "%~dp0"

echo ================================================================
echo Test WavePaint with ONNX Runtime (GPU Accelerated!)
echo ================================================================
echo.
echo This uses the ONNX file you already exported.
echo Expected: 5-10x faster than PyTorch!
echo.

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache
set PYTHONPATH=%~dp0python_packages
set PYTHONUNBUFFERED=1

echo Running test...
echo.

python -u test_wavepaint_onnx.py

pause
