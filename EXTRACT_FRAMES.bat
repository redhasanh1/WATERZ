@echo off
cd /d "%~dp0"

set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%PATH%

echo ======================================
echo Extract Frames from Sora Video
echo ======================================
echo.

REM Get video file
set /p VIDEO_PATH="Enter path to Sora video (drag and drop here): "

REM Remove quotes
set VIDEO_PATH=%VIDEO_PATH:"=%

if not exist "%VIDEO_PATH%" (
    echo Error: Video not found!
    pause
    exit /b
)

echo.
echo Extracting frames to NEW_SORA_TRAINING\images\
echo.

python extract_frames.py "%VIDEO_PATH%"

pause
