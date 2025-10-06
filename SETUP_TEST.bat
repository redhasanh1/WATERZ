@echo off
echo ================================================================
echo SETUP TEST VIDEO
echo ================================================================
echo.
echo To test watermark removal with audio, you need a test video.
echo.
echo STEP 1: Find a video file on your computer that has:
echo   - Audio (music, voice, etc)
echo   - A watermark (optional, but better for testing)
echo   - Any format: .mp4, .mov, .avi, etc
echo.
echo STEP 2: Copy that video file here:
echo   %CD%\uploads\
echo.
echo STEP 3: Rename it to: test_video.mp4
echo.
echo ================================================================
pause

REM Create uploads folder
if not exist uploads mkdir uploads

echo.
echo Opening uploads folder...
start explorer "%CD%\uploads"
echo.
echo ================================================================
echo INSTRUCTIONS:
echo ================================================================
echo.
echo 1. Copy a video file into the folder that just opened
echo 2. Rename it to: test_video.mp4
echo 3. Make sure it has audio! (play it to verify)
echo 4. Then run: RUN_FULL_TEST.bat
echo.
echo ================================================================
echo.
echo Current files in uploads folder:
dir /b uploads
echo.
if exist uploads\test_video.mp4 (
    echo [OK] test_video.mp4 found!
    echo.
    echo Checking if it has audio...
    ffprobe -v error -select_streams a:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 uploads\test_video.mp4 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [OK] Video has audio!
        echo.
        echo You're ready! Run: RUN_FULL_TEST.bat
    ) else (
        echo [WARNING] Could not detect audio
    )
) else (
    echo [WAITING] No test_video.mp4 found yet
    echo.
    echo Please copy a video file to: %CD%\uploads\test_video.mp4
)
echo.
pause
