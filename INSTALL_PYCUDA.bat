@echo off
cd /d "%~dp0"

echo ================================================================
echo Installing PyCUDA (Required for TensorRT)
echo ================================================================
echo.

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp

echo Installing PyCUDA...
echo.

python -m pip install pycuda --target python_packages

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS!
    echo ================================================================
    echo.
    echo PyCUDA installed
    echo Now run: EXPORT_WAVEPAINT_TENSORRT.bat
    echo.
) else (
    echo.
    echo ================================================================
    echo ERROR - PyCUDA installation failed
    echo ================================================================
    echo.
    echo PyCUDA can be tricky to install on Windows.
    echo.
    echo ALTERNATIVE: Use ONNX Runtime instead!
    echo Run: INSTALL_ONNXRUNTIME.bat
    echo Then: TEST_WAVEPAINT_ONNX.bat
    echo.
    echo ONNX Runtime gives 5-10x speedup without PyCUDA!
    echo.
)

pause
