@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Video FPS Reducer to 16 FPS
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

echo Reducing to 16 FPS (keeping original duration)...
echo.

REM Reduce FPS to 16 using ffmpeg
ffmpeg -i "!input_video!" -vf "fps=16" -c:v libx264 -preset fast -crf 23 test_16fps.mp4 -y -loglevel error

if %ERRORLEVEL%==0 (
    echo.
    echo Success! Created test_16fps.mp4
    echo   - Original duration kept
    echo   - 16 FPS reduced from original
    echo   - From: !input_video!
) else (
    echo.
    echo ERROR: ffmpeg failed
)

echo.
pause
