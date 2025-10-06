@echo off
cd /d "%~dp0"

set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%~dp0TensorRT-10.7.0.23\lib;%PATH%

python test_full_removal.py
pause
