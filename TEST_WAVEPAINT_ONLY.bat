@echo off
cd /d "%~dp0"

echo ================================================================
echo WavePaint Test - NO YOLO, NO TENSORRT
echo ================================================================
echo.
echo This will:
echo   1. Load your test image
echo   2. Let YOU draw a mask on the watermark
echo   3. Inpaint with LaMa
echo   4. Inpaint with WavePaint
echo   5. Save both for comparison
echo.

REM Force D drive temp/cache
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0cache
set XDG_CACHE_HOME=%~dp0cache
set OPENCV_TEMP_PATH=%~dp0temp
set PYTHONPATH=%~dp0python_packages
set PYTHONUNBUFFERED=1

echo Running test...
echo.

python -u test_wavepaint_only.py

echo.
pause
