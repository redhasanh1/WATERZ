@echo off
cd /d "%~dp0"

set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%~dp0TensorRT-10.7.0.23\lib;%PATH%

echo ========================================
echo FULL VIDEO WATERMARK REMOVAL TEST
echo ========================================
echo.
echo Technology Stack:
echo   - TensorRT YOLO (GPU accelerated detection)
echo   - CUDA (GPU acceleration)
echo   - LAMA AI Inpainting
echo   - 0px padding (tight detection)
echo.
echo This will:
echo   1. Pick a random video from videostotrain/
echo   2. Detect watermarks with TensorRT YOLO
echo   3. Remove with LAMA inpainting
echo   4. Save to test_video_removal_result.mp4
echo.
echo ========================================
echo.

python test_video_removal.py
pause
