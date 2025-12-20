@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo SEND VIDEO - Direct to 4090 Sender (SAM2 Masks)
echo ============================================================
echo   - Load video, draw bbox around watermark
echo   - Upload to B2 CDN
echo   - Submit to wsl_sam2_local queue (4090_sender)
echo ============================================================
echo.

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

"%PYTHON_PATH%" static_mask_gui.py

pause
