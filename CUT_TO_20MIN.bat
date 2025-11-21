@echo off
REM Cut any video to first 20 minutes for testing
REM Usage: CUT_TO_20MIN.bat input.mp4 output.mp4

if "%~1"=="" (
    echo Usage: CUT_TO_20MIN.bat input.mp4 output.mp4
    echo.
    echo Example: CUT_TO_20MIN.bat "C:\SurFastVideos\comedy.mp4" "C:\SurFastVideos\comedy_20min.mp4"
    exit /b 1
)

if "%~2"=="" (
    echo Usage: CUT_TO_20MIN.bat input.mp4 output.mp4
    exit /b 1
)

set INPUT=%~1
set OUTPUT=%~2

echo [CUT] Cutting first 20 minutes from: %INPUT%
echo [CUT] Output: %OUTPUT%

REM Cut to 20 minutes (1200 seconds) with GPU acceleration
ffmpeg -hwaccel cuda -i "%INPUT%" -t 1200 -c:v h264_nvenc -preset fast -b:v 5M -c:a copy "%OUTPUT%"

echo.
echo [OK] Done! First 20 minutes saved to: %OUTPUT%
echo [OK] Now you can test with: .\RUN_SAM2_WSL2_GUI.ps1 "%OUTPUT%"
