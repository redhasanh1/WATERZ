@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo Starting DUAL Celery workers (2x concurrency=2 = 4 parallel segments)...
echo.
echo ============================================================
echo DUAL WORKER MODE: 2 parallel workers @ concurrency=2 each
echo Expected: 15-20 seconds for 300-frame video (BEATS RTX 5070!)
echo ============================================================
echo.

REM Activate Visual Studio 2022 C++ environment (required for torch.compile)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo [OK] Visual Studio C++ environment activated
)

REM Ensure TensorRT DLLs and tools are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM BASELINE: Use PyTorch RAFT (NO optimizations!)
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=0

REM Disable RFCNet torch.compile (requires Triton which isn't available on Windows)
set RFCNET_TORCHTRT=0

REM Disable Flash Attention
set ENABLE_FLASH_ATTENTION=0

echo.
echo ============================================================
echo DUAL WORKER CONFIG:
echo   - Workers: 2 parallel workers (worker1 + worker2)
echo   - Concurrency: 2 per worker (4 total segments)
echo   - YOLO: TensorRT batch 128 RTX 4090 (800-1400 fps)
echo   - RAFT: PyTorch FP16 autocast (~10.5 it/s baseline)
echo   - Expected: 15-20 seconds (FASTER than RTX 5070's 20s!)
echo ============================================================
echo.

echo [LAUNCH] Starting Worker 1...
start "Celery Worker 1" cmd /k "cd /d "%~dp0" && "%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=2 --hostname=worker1@%%h"

echo [WAIT] Waiting 5 seconds before starting Worker 2...
timeout /t 5 /nobreak >nul

echo [LAUNCH] Starting Worker 2...
start "Celery Worker 2" cmd /k "cd /d "%~dp0" && "%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=2 --hostname=worker2@%%h"

echo.
echo ============================================================
echo [OK] 2 workers launched! Check the new console windows.
echo      Each worker will process 2 segments in parallel.
echo      Total: 4 segments can run simultaneously!
echo.
echo      Expected performance: 15-20 seconds for 300-frame video
echo      (30-50%% faster than RTX 5070's 20 seconds!)
echo ============================================================
echo.
echo Press any key to stop all workers and exit...
pause >nul

REM Kill all celery workers when user presses key
taskkill /F /FI "WINDOWTITLE eq Celery Worker 1*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Celery Worker 2*" >nul 2>&1
echo [CLEANUP] Workers stopped.
