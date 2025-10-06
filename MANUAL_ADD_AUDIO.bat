@echo off
echo ================================================================
echo MANUAL AUDIO MERGE
echo ================================================================
echo.
echo This will manually merge audio from original video
echo into the processed video (temp_no_audio.avi)
echo.

if not exist temp_no_audio.avi (
    echo [ERROR] temp_no_audio.avi not found!
    echo Run the test first to generate the video.
    pause
    exit /b 1
)

if not exist uploads\test_video.mp4 (
    echo [ERROR] uploads\test_video.mp4 not found!
    pause
    exit /b 1
)

echo [OK] Found temp_no_audio.avi
echo [OK] Found uploads\test_video.mp4
echo.

echo Running FFmpeg to merge audio...
echo.

ffmpeg -y -i temp_no_audio.avi -i uploads\test_video.mp4 -map 0:v:0 -map 1:a:0 -c:v libx264 -preset ultrafast -crf 18 -c:a aac -b:a 192k OUTPUT_WITH_AUDIO.mp4

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
