@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo SEND VIDEO LOCAL - Fully Local Pipeline (No B2 Uploads)
echo ============================================================
echo   - Load video, draw bbox around watermark
echo   - Keep masks LOCAL (no B2 upload)
echo   - Submit to wsl_sam2_local queue
echo ============================================================
echo.

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

"%PYTHON_PATH%" static_mask_gui_local.py

pause
