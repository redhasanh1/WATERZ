@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Move to repo root of this script (WATERZ folder)
cd /d "%~dp0"

REM Force D drive temp/cache (keep everything local to project)
set "PIP_CACHE_DIR=%~dp0pip_cache"
set "TEMP=%~dp0temp"
set "TMP=%~dp0temp"
set "TMPDIR=%~dp0temp"
set "TORCH_HOME=%~dp0cache"
set "XDG_CACHE_HOME=%~dp0cache"
set "OPENCV_TEMP_PATH=%~dp0temp"

REM Prefer vendored packages if present
set "PYTHONPATH=%~dp0python_packages;%~dp0web\python_packages"
set "PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%~dp0TensorRT-10.7.0.23\lib;%PATH%"

REM Configure Redis broker (override here if needed)
if "%REDIS_URL%"=="" (
  set "REDIS_URL=redis://:watermarkz_secure_2024@6.tcp.ngrok.io:11553/0"
)

echo ================================================================
echo WatermarkAI - Production API (Waitress)
echo ================================================================
echo Using PYTHONPATH: %PYTHONPATH%
echo Using REDIS_URL : %REDIS_URL%
REM Auto-load public base URL for workers (optional)
if "%TUNNEL_URL%"=="" (
  if exist "web\tunnel_url.txt" (
    for /f "usebackq delims=" %%i in ("web\tunnel_url.txt") do set "TUNNEL_URL=%%i"
  )
)
if not "%TUNNEL_URL%"=="" (
  set "API_BASE_URL=%TUNNEL_URL%"
  set "TEMP_BASE_URL=%TUNNEL_URL%"
  echo Using TUNNEL_URL : %TUNNEL_URL%
)
echo.

REM Activate venv if present
if exist venv\Scripts\activate.bat (
  call venv\Scripts\activate.bat
)

REM Ensure waitress is installed (Windows-friendly WSGI server)
python -c "import waitress" 1>nul 2>nul
if errorlevel 1 (
  echo Installing waitress...
  python -m pip install --upgrade pip >nul
  python -m pip install waitress
)

REM Disable Python buffering and mark production
set "PYTHONUNBUFFERED=1"
set "FLASK_ENV=production"

REM Run the WEB server module (web\server_production.py) with Waitress
cd /d "%~dp0web"
echo Starting production server on http://0.0.0.0:9000 ...
python -u -m waitress --host=0.0.0.0 --port=9000 --threads=4 server_production:app

REM If we get here, waitress exited
echo.
echo [ERROR] Server stopped unexpectedly.
pause
endlocal
