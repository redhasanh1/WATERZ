@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

echo ============================================================
echo SAM2 10FPS PIPELINE - Ultra-Fast Mask Generation
echo ============================================================
echo.
echo This pipeline:
echo   1. Detects video FPS (30fps, 60fps, etc.)
echo   2. Converts to 10fps using NVDEC/NVENC
echo   3. Runs SAM2 hybrid (TensorRT + torch.compile)
echo   4. Outputs 10fps masks + mapping for zero-copy expansion
echo.
echo Speedup: 6x for 60fps, 3x for 30fps
echo.
echo ============================================================

"%PYTHON_PATH%" sam2_10fps_pipeline.py %*

pause
