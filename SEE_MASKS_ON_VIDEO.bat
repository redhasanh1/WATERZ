@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

echo ============================================================
echo SAM2 Mask Visualization - See masks overlaid on video
echo ============================================================
echo.
echo This will:
echo   1. Read frames from temp_sam2_frames
echo   2. Read masks from temp_sam2_masks
echo   3. Create visualization video with green mask overlay
echo   4. Open the result automatically
echo.
echo ============================================================

"%PYTHON_PATH%" visualize_masks_on_video.py

pause
