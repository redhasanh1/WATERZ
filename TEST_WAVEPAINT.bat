@echo off
cd /d "%~dp0"

echo ================================================================
echo WavePaint vs LaMa Inpainting Test
echo ================================================================
echo.
echo This script will:
echo   1. Load your trained YOLO watermark detector
echo   2. Detect watermarks in a test image
echo   3. Inpaint using LaMa (current method)
echo   4. Inpaint using WavePaint (new method)
echo   5. Save results for comparison
echo.
echo ================================================================
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

echo Current directory: %cd%
echo Python packages: %PYTHONPATH%
echo.

REM Check if WavePaint weights exist
if not exist "wavepaint\WavePaint_blocks4_dim128_modules8_celebhq256.pth" (
    echo ================================================================
    echo WARNING: WavePaint weights not found!
    echo ================================================================
    echo.
    echo WavePaint requires pretrained model weights.
    echo.
    echo Download from:
    echo https://github.com/pranavphoenix/WavePaint/releases
    echo.
    echo Place the .pth file in: wavepaint\
    echo.
    echo For now, the test will only run LaMa.
    echo.
    pause
)

echo Running test...
echo.

python -u test_wavepaint.py

echo.
echo ================================================================
echo Test Complete!
echo ================================================================
echo.
echo Check the results in: wavepaint_test\results\
echo.
echo Files:
echo   01_original.jpg      - Input image with watermark
echo   02_mask.png          - YOLO detection mask
echo   03_lama_result.jpg   - LaMa result (current method)
echo   04_wavepaint_result.jpg - WavePaint result (if weights available)
echo.
echo Compare the quality visually to see which is better!
echo.
pause
