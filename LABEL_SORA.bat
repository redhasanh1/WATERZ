@echo off
cd /d "%~dp0"

set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%PATH%

echo ======================================
echo Auto-Label Sora Watermarks
echo ======================================
echo.

python auto_label_sora.py

pause
