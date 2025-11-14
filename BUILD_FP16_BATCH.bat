@echo off
cd /d "%~dp0"

REM Try to find Python with CUDA torch
set PYTHON_CMD=python

REM Try Python 3.12 direct path first (most reliable on Windows)
set "PY312_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY312_PATH%" (
  set "PYTHON_CMD=%PY312_PATH%"
  echo Using Python 3.12
  goto :check_cuda
)

REM Try py launcher with Python 3.12
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
  set PYTHON_CMD=py -3.12
  echo Using Python 3.12 via launcher
  goto :check_cuda
)

REM Try venv if py -3.12 not found
if exist venv\Scripts\activate.bat (
  echo Activating venv...
  call venv\Scripts\activate.bat
)

:check_cuda

REM Check if current Python has CUDA torch
"%PYTHON_CMD%" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo.
  echo [ERROR] Python does not have CUDA torch installed!
  echo Current Python:
  "%PYTHON_CMD%" --version
  echo.
  echo You need Python with CUDA PyTorch to build TensorRT engines.
  echo.
  echo Install CUDA PyTorch in Python 3.12:
  echo   py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  echo   py -3.12 -m pip install tensorrt ultralytics
  echo.
  pause
  exit /b 1
)

echo ============================================================
echo Building FP16 TensorRT Batch Engine (RTX 4090)
echo Supports batch 1-256 for EXTREME SPEED (2x RTX 5070)
echo ============================================================
echo.

echo [*] Building FP16 batch engine (3-5 minutes)...
echo.

"%PYTHON_CMD%" build_fp16_batch_engine.py

if errorlevel 1 (
    echo.
    echo [!] Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! RTX 4090 FP16 batch engine ready!
echo ============================================================
echo.
echo Next: Run BENCHMARK_FP16_BATCH.bat to test performance
echo Expected: 0.7-1.2ms per frame with batch 128 (800-1400 fps)
echo.
pause
