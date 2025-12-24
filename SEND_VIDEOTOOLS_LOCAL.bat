@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo VIDEO TOOLS - Local GPU Processing
echo ============================================================
echo   Operations:
echo   - Keep Object (remove background)
echo   - Remove Object (keep background)
echo   - Fill Inside/Outside with Color
echo   - Blur Inside/Outside
echo.
echo   Export: MP4 (NVENC) / WebM (Alpha) / PNG Sequence
echo ============================================================
echo.

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Local mode - no B2 uploads
set SKIP_B2_UPLOAD=1
set USE_DALI_STREAMING=1
set SAM2_MAX_HEIGHT=480

"%PYTHON_PATH%" video_tools_gui.py

pause
