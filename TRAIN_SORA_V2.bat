@echo off
cd /d "%~dp0"

set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%PATH%

REM Use Python 3.12 with CUDA torch
set PYTHON_CMD=py -3.12

REM Check if Python 3.12 exists
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.12 not found!
  pause
  exit /b 1
)

REM Check CUDA torch
%PYTHON_CMD% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.12 does not have CUDA torch!
  pause
  exit /b 1
)

echo ======================================
echo Train YOLO v2 on Sora Watermarks
echo Dataset: yolo_training (4,481 frames)
echo RTX 4090 - Batch Size 16
echo ======================================
echo.

%PYTHON_CMD% train_sora_v2.py

echo.
echo ======================================
echo Training complete! Now run:
echo EXPORT_SORA_V2_TENSORRT.bat
echo ======================================
pause
