@echo off
echo Extracting test frame...
cd /d "%~dp0"

REM Force everything to D drive
set PYTHONPATH=%~dp0python_packages
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0pip_cache
set XDG_CACHE_HOME=%~dp0pip_cache
mkdir temp 2>nul

python test_image.py

echo.
echo Installing IOPaint for image watermark removal...
python -m pip install --target python_packages --cache-dir pip_cache --no-warn-script-location iopaint

echo.
echo Starting IOPaint web interface...
echo Open http://localhost:8080 in your browser
echo Load test_frame.jpg and manually paint over the watermark
echo.
set PYTHONPATH=%~dp0python_packages
python -m iopaint start --model=lama --device=cuda --port=8080

pause
