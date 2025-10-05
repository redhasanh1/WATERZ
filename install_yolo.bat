@echo off
echo Installing YOLOv8 and dependencies...

cd /d "%~dp0"

REM Force ALL temp/cache to D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0pip_cache
set XDG_CACHE_HOME=%~dp0pip_cache

mkdir temp 2>nul
mkdir pip_cache 2>nul

echo.
echo Installing ultralytics (YOLOv8)...
python -m pip install ultralytics --target python_packages

echo.
echo Installing simple-lama-inpainting...
python -m pip install simple-lama-inpainting --target python_packages

echo.
echo Done!
pause
