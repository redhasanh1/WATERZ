@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Batch Video Cutter - 3 Second Clips
echo ============================================
echo.

set "input_folder=videostotrain"
set "output_folder=."
set /a count=0

echo Processing videos from %input_folder%...
echo Output: 3-second clips in %output_folder%
echo.

REM Process each MP4 video
for %%f in (%input_folder%\*.mp4) do (
    set /a count+=1
    set "input=%%f"
    set "filename=%%~nf"

    echo [!count!] Processing: %%~nxf

    REM Generate random start time (0-10 seconds)
    set /a "start=%RANDOM% %% 11"

    echo    - Cutting from !start! seconds

    REM Cut to 3 seconds and save to root folder
    ffmpeg -ss !start! -i "!input!" -t 3 -c copy "%output_folder%\!filename!_3sec.mp4" -y -loglevel error

    if !ERRORLEVEL!==0 (
        echo    - ✓ Created: !filename!_3sec.mp4
    ) else (
        echo    - ✗ Failed to process
    )
    echo.
)

if %count%==0 (
    echo ERROR: No MP4 videos found in %input_folder%
) else (
    echo ============================================
    echo Completed! Processed %count% videos
    echo All 3-second clips saved to: %output_folder%
    echo ============================================
)

echo.
pause
