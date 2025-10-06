@echo off
echo ================================================================
echo COMPLETE WATERMARK REMOVAL + AUDIO TEST
echo ================================================================
echo.
echo This test uses the EXACT same code as your production server
echo.
pause

REM Activate venv if exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Running test...
echo.

python test_watermark_with_audio.py

echo.
echo ================================================================
echo TEST COMPLETE
echo ================================================================
echo.
echo Check the output above for:
echo   [PASS] or [FAIL] messages
echo.
echo If successful, play: results\final_WITH_AUDIO.mp4
echo.
pause
