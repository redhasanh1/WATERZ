@echo off
REM Convert video to 7fps (full duration) for SAM2 testing
REM Usage: Drop a video file onto this batch file, or run it and enter filename

setlocal enabledelayedexpansion

echo ============================================================
echo    SAM2 Video Converter: 7fps (Full Duration)
echo ============================================================
echo.

REM Check if video was dragged onto the batch file
if "%~1"=="" (
    set /p "INPUT_VIDEO=Enter video filename (in this folder): "
) else (
    set "INPUT_VIDEO=%~1"
)

REM Check if file exists
if not exist "!INPUT_VIDEO!" (
    echo [ERROR] File not found: !INPUT_VIDEO!
    pause
    exit /b 1
)

REM Get the base filename without extension
for %%F in ("!INPUT_VIDEO!") do (
    set "FILENAME=%%~nF"
    set "EXT=%%~xF"
)

REM Set output filename
set "OUTPUT=!FILENAME!_7fps.mp4"

echo.
echo [INPUT]  !INPUT_VIDEO!
echo [OUTPUT] !OUTPUT!
echo [PARAMS] 7fps, full duration, scale to 720p (SAM2 optimized)
echo.

REM Convert using Python script
cd /d "%~dp0"
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" convert_video_7fps.py "!INPUT_VIDEO!"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] Video converted!
    echo ============================================================
    echo.
    echo File: !OUTPUT!
    echo.
    echo Ready for SAM2 interactive testing!
    echo ============================================================
) else (
    echo.
    echo [ERROR] Conversion failed!
)

echo.
pause
