@echo off
cd /d "%~dp0"

REM Force D drive temp/cache
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0cache
set XDG_CACHE_HOME=%~dp0cache
set OPENCV_TEMP_PATH=%~dp0temp
set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%~dp0TensorRT-10.7.0.23\lib;%PATH%

echo ================================================================
echo Starting PRODUCTION Flask server with Waitress
echo ================================================================
echo Using packages from: %PYTHONPATH%
echo.

REM Activate venv if exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Check if Waitress is installed
python -c "import waitress" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Waitress not installed!
    echo Run INSTALL_WAITRESS.bat first
    pause
    exit /b 1
)

echo Starting production server on port 9000...
echo.

REM Disable Python output buffering to prevent hanging
set PYTHONUNBUFFERED=1

REM Start with Waitress (production WSGI server)
REM Use python -m to run waitress (doesn't need PATH)
python -u -m waitress --host=0.0.0.0 --port=9000 --threads=4 server_production:app

REM If we get here, something failed
echo.
echo [ERROR] Server stopped unexpectedly
pause
