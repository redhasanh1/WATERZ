@echo off
cd /d "%~dp0"

set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%PATH%

REM Try to find Python with CUDA torch
set PYTHON_CMD=python

REM Try Python 3.12 first (has CUDA torch support)
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
  set PYTHON_CMD=py -3.12
  echo Using Python 3.12
)

REM Try venv if py -3.12 not found
if exist venv\Scripts\activate.bat (
  if "%PYTHON_CMD%"=="python" (
    echo Activating venv...
    call venv\Scripts\activate.bat
  )
)

REM Check if current Python has CUDA torch
%PYTHON_CMD% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo.
  echo [ERROR] Python does not have CUDA torch installed!
  echo Current Python:
  %PYTHON_CMD% --version
  echo.
  echo You need Python with CUDA PyTorch to train on GPU.
  echo.
  echo Install CUDA PyTorch in Python 3.12:
  echo   py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  echo   py -3.12 -m pip install ultralytics
  echo.
  pause
  exit /b 1
)

echo ======================================
echo Train YOLO on NEW Sora Watermarks
echo RTX 4090 - Batch Size 16
echo ======================================
echo.

%PYTHON_CMD% train_new_sora.py

pause
