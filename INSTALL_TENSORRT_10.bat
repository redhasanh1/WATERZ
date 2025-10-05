@echo off
echo ============================================================
echo Installing TensorRT 10.7 + cuDNN 9.13.1 (CUDA 12.1)
echo ============================================================
echo.
pause

cd /d "%~dp0"

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache

echo.
echo Step 1: Installing TensorRT 10.7 Python wheel...
python -m pip install --target python_packages "TensorRT-10.7.0.23\python\tensorrt-10.7.0-cp311-none-win_amd64.whl"

echo.
echo Step 2: Copying TensorRT DLLs to python_packages...
copy "TensorRT-10.7.0.23\lib\*.dll" python_packages\

echo.
echo Step 3: Copying cuDNN 9.13.1 DLLs to python_packages...
copy "cudnn-windows-x86_64-9.13.1.26_cuda12-archive\bin\*.dll" python_packages\

echo.
echo Step 4: Testing TensorRT installation...
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%PATH%
python -c "import sys; sys.path.insert(0, 'python_packages'); import tensorrt as trt; print(''); print('='*60); print('TensorRT version:', trt.__version__); print('='*60); print('')"

echo.
echo If you see TensorRT version 10.7.0 above, you're ready!
echo.
pause
