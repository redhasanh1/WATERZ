@echo off
echo ================================================================
echo MANUAL AUDIO MERGE
echo ================================================================
echo.
echo This will manually merge audio from original video
echo into the processed video (temp_no_audio.avi)
echo.

REM Check for temp file (could be .avi or .mp4)
set TEMP_FILE=
if exist temp_no_audio.avi set TEMP_FILE=temp_no_audio.avi
if exist temp_no_audio.mp4 set TEMP_FILE=temp_no_audio.mp4

if "%TEMP_FILE%"=="" (
    echo [ERROR] Temp video not found!
    echo Looking for: temp_no_audio.avi or temp_no_audio.mp4
    echo Run the test first to generate the video.
    pause
    exit /b 1
)

if not exist uploads\test_video.mp4 (
    echo [ERROR] uploads\test_video.mp4 not found!
    pause
    exit /b 1
)

echo [OK] Found %TEMP_FILE%
echo [OK] Found uploads\test_video.mp4
echo.

echo Running FFmpeg to merge audio...
echo.

ffmpeg -y -i %TEMP_FILE% -i uploads\test_video.mp4 -map 0:v:0 -map 1:a:0 -c:v libx264 -preset ultrafast -crf 18 -c:a aac -b:a 192k OUTPUT_WITH_AUDIO.mp4

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS!
    echo ================================================================
    echo.
    echo Audio merged successfully!
    echo Output: OUTPUT_WITH_AUDIO.mp4
    echo.

    if exist OUTPUT_WITH_AUDIO.mp4 (
        echo Verifying audio...
        ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 OUTPUT_WITH_AUDIO.mp4 2>nul | findstr audio >nul
        if %ERRORLEVEL% EQU 0 (
            echo [OK] Output has audio!
        ) else (
            echo [WARNING] Could not detect audio
        )

        echo.
        echo Opening file location...
        start explorer /select,"%CD%\OUTPUT_WITH_AUDIO.mp4"
    )
) else (
    echo.
    echo ================================================================
    echo FAILED!
    echo ================================================================
    echo.
    echo FFmpeg failed with error code: %ERRORLEVEL%
    echo.
    echo Check above for error messages.
)

echo.
pause
