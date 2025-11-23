@echo off
echo ================================================================================
echo TORCH-TENSORRT AOT MODEL COMPILATION (Windows)
echo ================================================================================
echo.
echo This will pre-compile RAFT and RFCNet models with torch-tensorrt.
echo Compilation takes ~60-120 seconds.
echo.
echo Benefits:
echo   - Workers load models in 0.5s (not 30-60s per worker)
echo   - Enables true 4x parallel speedup
echo   - One-time compilation, reuse forever
echo.

cd /d "%~dp0"

REM Check if torch-tensorrt is installed
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -c "import torch_tensorrt" 2>nul
if errorlevel 1 (
    echo [ERROR] torch-tensorrt is not installed!
    echo.
    echo Install with:
    echo   pip install torch-tensorrt --index-url https://download.pytorch.org/whl/cu124
    echo.
    pause
    exit /b 1
)

echo [OK] torch-tensorrt is installed
echo.

REM Configuration
SET TORCH_TENSORRT_PRECISION=fp16
SET TORCH_TENSORRT_WORKSPACE=4096

echo [CONFIG] Precision: %TORCH_TENSORRT_PRECISION%
echo [CONFIG] Workspace: %TORCH_TENSORRT_WORKSPACE% MB
echo.

REM Run compilation script
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" build_torch_tensorrt_models.py

if errorlevel 1 (
    echo.
    echo [ERROR] Compilation failed! Check errors above.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Models compiled successfully.
echo ================================================================================
echo.
echo To use compiled models:
echo   1. Set: USE_TORCH_TENSORRT=1
echo   2. Run your script normally
echo.
echo For parallel processing:
echo   SET MAX_PARALLEL_STREAMS=4
echo   SET PARALLEL_MODE=multiprocessing
echo   SET USE_TORCH_TENSORRT=1
echo.
pause
