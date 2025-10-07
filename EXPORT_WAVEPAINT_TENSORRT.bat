@echo off
cd /d "%~dp0"

echo ================================================================
echo Export WavePaint to TensorRT (10-20x Speedup!)
echo ================================================================
echo.
echo This will convert WavePaint from:
echo   BEFORE: 8 seconds/frame  (0.125 fps)
echo   AFTER:  0.5 seconds/frame (2 fps)
echo.
echo This takes 2-5 minutes. Please wait...
echo.

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache
set PYTHONPATH=%~dp0python_packages
set PYTHONUNBUFFERED=1

REM Add TensorRT DLLs to PATH (CRITICAL!)
set PATH=%~dp0python_packages;%PATH%

echo Running export...
echo.

python -u export_wavepaint_tensorrt.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS!
    echo ================================================================
    echo.
) else (
    echo.
    echo ================================================================
    echo EXPORT FAILED
    echo ================================================================
    echo.
)

pause
