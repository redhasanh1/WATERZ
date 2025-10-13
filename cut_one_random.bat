@echo off
setlocal enabledelayedexpansion

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

REM Generate random start time (0-10 seconds)
set /a "start=%RANDOM% %% 11"

echo Cutting 3 seconds from !input_video! at %start%s...

REM Cut video using ffmpeg
ffmpeg -ss %start% -i "!input_video!" -t 3 -c copy test_3sec.mp4 -y -loglevel error

if %ERRORLEVEL%==0 (
    echo Done! Created test_3sec.mp4
) else (
    echo ERROR: ffmpeg failed
    pause
)
