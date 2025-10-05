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

echo Starting Flask server...
echo Using packages from: %PYTHONPATH%
echo.
python server_production.py
pause 
