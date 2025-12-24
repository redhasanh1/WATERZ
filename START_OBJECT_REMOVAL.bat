@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   OBJECT REMOVAL SERVER
echo ============================================================
echo   Local Flask server for SAM2 object tracking + export
echo
echo   Features:
echo   - Upload video
echo   - Click to select object (SAM2 TensorRT)
echo   - Track through full video
echo   - Export with background options (transparent, color, blur)
echo   - Export formats: MP4, WebM
echo ============================================================
echo.

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

REM Install dependencies if needed
echo [DEPS] Checking dependencies...
pip show flask >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [DEPS] Installing Flask...
    pip install flask flask-cors
)

echo.
echo [SERVER] Starting on http://localhost:5000
echo [SERVER] Press Ctrl+C to stop
echo.

python object_removal_server.py

pause
