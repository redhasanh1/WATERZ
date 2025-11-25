@echo off
REM ============================================================================
REM RTX 3070 Processing Worker - Celery Worker
REM ============================================================================
REM
REM This runs on local 3070 machines and handles:
REM   - Pulling segment jobs from queue
REM   - Running ProPainter inpainting
REM   - Uploading results back
REM
REM Queue: processing
REM Concurrency: 1 (one segment per GPU at a time)
REM ============================================================================

echo ============================================================
echo   RTX 3070 Processing Worker
echo ============================================================
echo.

REM Change to script directory
cd /d %~dp0

REM Set Python path
set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

REM Fallback to system Python if local not found
if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo Using Python: %PYTHON_EXE%
echo.

REM GPU Optimizations for RTX 3070
echo Setting RTX 3070 optimizations...
set ENABLE_FLASH_ATTENTION=0
set USE_TORCH_COMPILE=0
set TORCH_CUDAGRAPHS=0
set TORCHINDUCTOR_CUDAGRAPHS=0

REM Optional: Set specific GPU if multiple GPUs
REM set CUDA_VISIBLE_DEVICES=0

echo.

REM Start Celery worker
echo Starting processing worker...
echo Queue: processing
echo Concurrency: 1
echo.

%PYTHON_EXE% -m celery -A celery_3070_worker worker ^
    --queues=processing ^
    --concurrency=1 ^
    --loglevel=info ^
    --pool=solo ^
    --hostname=worker3070@%%h

pause
