@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

echo ============================================================
echo SAM2 Mask Visualization - With 10fps Expansion
echo ============================================================
echo.
echo This will:
echo   1. Read mask_mapping.json for FPS multiplier
echo   2. Read masks from temp_sam2_masks
echo   3. Expand 10fps masks to original video FPS
echo   4. Create visualization with green overlay at correct speed
echo.
echo Usage: SEE_MASKS_ON_VIDEO.bat [video_path]
echo   - If no video provided, file picker will open
echo.
echo ============================================================

"%PYTHON_PATH%" visualize_masks_on_video.py %*

pause
