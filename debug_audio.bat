@echo off
echo ================================================================
echo AUDIO DEBUG TEST
echo ================================================================
echo.
echo This will help us figure out why audio isn't working
echo.

REM 1. Check FFmpeg
echo [1/5] Checking FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    [FAIL] FFmpeg not found in PATH!
    echo.
    echo    Download: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
    echo    Extract to C:\ffmpeg
    echo    Add C:\ffmpeg\bin to PATH
    pause
    exit /b 1
)
echo    [PASS] FFmpeg found
ffmpeg -version | findstr "ffmpeg version"

REM 2. Check FFprobe
echo.
echo [2/5] Checking FFprobe...
where ffprobe >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    [FAIL] FFprobe not found!
    pause
    exit /b 1
)
echo    [PASS] FFprobe found

REM 3. Check test video exists
echo.
echo [3/5] Checking test video...
if not exist uploads\test_video.mp4 (
    echo    [FAIL] uploads\test_video.mp4 not found!
    echo.
    echo    Please place a video file at: uploads\test_video.mp4
    pause
    exit /b 1
)
echo    [PASS] uploads\test_video.mp4 exists

REM 4. Check if video has audio
echo.
echo [4/5] Checking if test video has audio...
ffprobe -v error -select_streams a:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 uploads\test_video.mp4 > audio_check.txt 2>&1
set /p AUDIO_PACKETS=<audio_check.txt
del audio_check.txt

if "%AUDIO_PACKETS%"=="" (
    echo    [FAIL] Video has NO audio track!
    echo.
    echo    Your test video doesn't have audio.
    echo    Please use a different video that has audio.
    pause
    exit /b 1
)

if "%AUDIO_PACKETS%"=="0" (
    echo    [FAIL] Video has audio track but no packets!
    pause
    exit /b 1
)

echo    [PASS] Video has audio track with %AUDIO_PACKETS% packets

REM 5. Test simple audio copy
echo.
echo [5/5] Testing simple audio extraction and merge...
if not exist results mkdir results

echo    Step A: Extract audio from test video...
ffmpeg -y -v error -i uploads\test_video.mp4 -vn -acodec copy results\extracted_audio.aac 2>error.log
if %ERRORLEVEL% NEQ 0 (
    echo    [FAIL] Audio extraction failed!
    type error.log
    del error.log
    pause
    exit /b 1
)
echo    [PASS] Audio extracted to results\extracted_audio.aac

echo.
echo    Step B: Test merging video + audio...
REM Create simple test: copy video and audio from same source
ffmpeg -y -v error -i uploads\test_video.mp4 -i uploads\test_video.mp4 -map 0:v:0 -map 1:a:0 -c:v libx264 -preset ultrafast -crf 23 -c:a aac -b:a 192k results\test_merged.mp4 2>error.log
if %ERRORLEVEL% NEQ 0 (
    echo    [FAIL] Audio merge failed!
    type error.log
    del error.log
    pause
    exit /b 1
)
echo    [PASS] Merged to results\test_merged.mp4

echo.
echo    Step C: Verify output has audio...
ffprobe -v error -select_streams a:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 results\test_merged.mp4 > verify.txt 2>&1
set /p OUTPUT_AUDIO=<verify.txt
del verify.txt

if "%OUTPUT_AUDIO%"=="" (
    echo    [FAIL] Output has NO audio!
    echo.
    echo    FFmpeg merge failed. Check FFmpeg installation.
    pause
    exit /b 1
)

echo    [PASS] Output has audio with %OUTPUT_AUDIO% packets!

echo.
echo ================================================================
echo ALL TESTS PASSED!
echo ================================================================
echo.
echo FFmpeg is working correctly and can merge audio.
echo.
echo Test file created: results\test_merged.mp4
echo Play this file - it should have audio!
echo.
echo If this file has audio but your watermark removal doesn't,
echo the issue is in the Python processing code.
echo.
pause

REM Now show what the actual command would be
echo.
echo ================================================================
echo ACTUAL FFMPEG COMMAND USED IN SERVER:
echo ================================================================
echo.
echo ffmpeg -y ^
echo   -i PROCESSED_VIDEO.avi ^
echo   -i ORIGINAL_VIDEO.mp4 ^
echo   -map 0:v:0 ^
echo   -map 1:a:0 ^
echo   -c:v libx264 ^
echo   -preset ultrafast ^
echo   -crf 18 ^
echo   -c:a aac ^
echo   -b:a 192k ^
echo   -strict experimental ^
echo   OUTPUT.mp4
echo.
pause
