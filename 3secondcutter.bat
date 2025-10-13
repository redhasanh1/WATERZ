@echo off
setlocal enabledelayedexpansion

echo ============================================
echo 3-Second Video Cutter
echo ============================================
echo.

REM Get a random video from videostotrain folder
set "folder=videostotrain"
set /a count=0

REM Count videos
for %%f in (%folder%\*.mp4) do (
    set /a count+=1
    set "video!count!=%%f"
)

if %count%==0 (
    echo ERROR: No videos found in %folder%
    pause
    exit /b 1
)

REM Pick random video
set /a "rand=%RANDOM% %% count + 1"
set "input_video=!video%rand%!"

echo Found %count% videos
echo Selected: !input_video!
echo.

REM Generate random start time (0-10 seconds)
set /a "start=%RANDOM% %% 11"

echo Cutting 3 seconds starting at %start% seconds...
echo.

REM Cut video using ffmpeg
ffmpeg -ss %start% -i "!input_video!" -t 3 -c copy test_3sec.mp4 -y

if %ERRORLEVEL%==0 (
    echo.
    echo ✓ Success! Created test_3sec.mp4
    echo   - 3 seconds long
    echo   - Starting at %start% seconds
    echo   - From: !input_video!
) else (
    echo.
    echo ERROR: ffmpeg failed
)

echo.
pause
