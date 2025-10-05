@echo off
cd /d "%~dp0"

set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%PATH%

echo ======================================
echo Extract Frames from Sora Videos
echo ======================================
echo.
echo Extracting 30 frames per video from videostotrain\ folder...
echo.

python extract_frames.py

pause
