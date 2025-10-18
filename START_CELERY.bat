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

REM Add TensorRT DLLs to PATH
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%~dp0TensorRT-10.7.0.23\lib;%PATH%

REM Redis URL is now auto-loaded by server_production.py from redis_url.txt
REM No need to set REDIS_URL environment variable - Python code handles it

echo Starting Celery worker (GPU processing)...
echo Using packages from: %PYTHONPATH%
echo Note: Redis URL auto-loaded from redis_url.txt by Python
echo.

REM Disable Python output buffering to prevent hanging
set PYTHONUNBUFFERED=1

python -u -m celery -A server_production.celery worker --loglevel=info --pool=solo
pause 
