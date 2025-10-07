@echo off
cd /d "%~dp0"

echo ================================================================
echo Installing ONNX Runtime GPU (TensorRT Alternative)
echo ================================================================
echo.
echo ONNX Runtime with CUDA is 5-10x faster than PyTorch
echo (Almost as fast as TensorRT, but no DLL issues!)
echo.

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp

echo Installing ONNX Runtime GPU...
echo.

python -m pip install onnxruntime-gpu --target python_packages

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS!
    echo ================================================================
    echo.
    echo ONNX Runtime GPU installed
    echo Now run: TEST_WAVEPAINT_ONNX.bat
    echo.
) else (
    echo.
    echo ================================================================
    echo ERROR
    echo ================================================================
    echo.
)

pause
