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
 
echo Starting Flask server... 
python server_production.py 
