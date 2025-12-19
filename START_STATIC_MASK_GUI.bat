@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo MASK GUI - Static + Tracked Modes
echo ============================================================
echo   - Draw shapes on video frames (rect, ellipse, polygon, brush)
echo   - STATIC: GPU generates mask, same mask all frames (fast!)
echo   - TRACKED: SAM2 detects + tracks object across frames
echo ============================================================
echo.
echo REQUIREMENTS:
echo   - Redis must be running
echo   - 4090_receiver.bat (ProPainter) for both modes
echo   - 4090_sender.bat (SAM2) for TRACKED mode only
echo.

REM Use explicit Python path
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
)
if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=python
)

echo Starting Mask GUI...
echo.

REM Add current directory to Python path
set PYTHONPATH=%~dp0;%PYTHONPATH%

"%PYTHON_PATH%" "%~dp0static_mask_gui.py"

pause
