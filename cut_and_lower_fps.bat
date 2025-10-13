@echo off
setlocal enabledelayedexpansion

echo ============================================
echo 3-Second Video Cutter + FPS Reducer
echo ============================================
echo.

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

echo Cutting 3 seconds starting at %start%s and reducing to 16 FPS...
echo.

REM Cut video and reduce FPS to 16 using ffmpeg
ffmpeg -ss %start% -i "!input_video!" -t 3 -vf "fps=16" -c:v libx264 -preset fast -crf 23 test_3sec_16fps.mp4 -y -loglevel error

if %ERRORLEVEL%==0 (
    echo.
    echo ✓ Success! Created test_3sec_16fps.mp4
    echo   - 3 seconds long
    echo   - 16 FPS (reduced from original)
    echo   - Starting at %start% seconds
    echo   - From: !input_video!
) else (
    echo.
    echo ERROR: ffmpeg failed
)

echo.
pause
