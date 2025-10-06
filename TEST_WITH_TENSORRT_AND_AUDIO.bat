@echo off
echo ================================================================
echo WATERMARK REMOVAL TEST
echo TensorRT + CUDA + LaMa + AUDIO PRESERVATION
echo ================================================================
echo.
echo This uses the EXACT same setup as production:
echo   - TensorRT YOLO (GPU accelerated detection)
echo   - Optimized LaMa (FP16 + CUDA inpainting)
echo   - FFmpeg audio merge
echo.
pause

REM Check test video
if not exist uploads\test_video.mp4 (
    echo [ERROR] Test video not found!
    echo.
    echo Run SETUP_TEST.bat first to add a test video
    pause
    exit /b 1
)

echo [OK] Test video found
echo.

REM Activate venv
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Starting test...
echo ================================================================
echo.

python test_video_removal_WITH_AUDIO.py

echo.
echo ================================================================
echo TEST COMPLETE
echo ================================================================
echo.
echo Output file: OUTPUT_WITH_AUDIO.mp4
echo Location: %CD%\OUTPUT_WITH_AUDIO.mp4
echo.
if exist OUTPUT_WITH_AUDIO.mp4 (
    echo [SUCCESS] File created!
    echo.
    echo Opening folder...
    start explorer /select,"%CD%\OUTPUT_WITH_AUDIO.mp4"
    echo.
    echo Play OUTPUT_WITH_AUDIO.mp4 to verify:
    echo   1. Watermark is removed
    echo   2. Audio is preserved
) else (
    echo [ERROR] File not found!
    echo Check the output above for errors.
)
echo.
pause
