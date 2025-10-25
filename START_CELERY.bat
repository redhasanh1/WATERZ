@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8 
 
REM Force D drive temp/cache 
set PIP_CACHE_DIR=%~dp0pip_cache 
set TEMP=%~dp0temp 
set TMP=%~dp0temp 
set TMPDIR=%~dp0temp 
set TORCH_HOME=%~dp0cache 
set XDG_CACHE_HOME=%~dp0cache 
set OPENCV_TEMP_PATH=%~dp0temp 
 
echo Starting Celery worker (GPU processing)... 

REM Ensure local python_packages are importable
set PYTHONPATH=%~dp0python_packages;%PYTHONPATH%

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Use module form so we don't rely on a celery.exe script
REM Use solo pool for Windows stability (avoids prefork errors)
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=solo 
