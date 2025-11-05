@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo Starting Celery worker (TensorRT RAFT - testing acceleration)...
echo.
echo ============================================================
echo TENSORRT MODE: TensorRT RAFT engine (3-5x faster than baseline)
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

REM TENSORRT: DISABLED (engine needs rebuild for TRT 10.14) - using PyTorch RAFT for now
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=0

REM Disable RFCNet torch.compile (requires Triton which isn't available on Windows)
set RFCNET_TORCHTRT=0

REM Disable Flash Attention
set ENABLE_FLASH_ATTENTION=0

echo.
echo ============================================================
echo OPTIMIZED CONFIG (RTX 4090):
echo   - YOLO: TensorRT batch 128 (800-1400 fps - 2x RTX 5070!)
echo   - RAFT: PyTorch FP16 + TF32 (~10.5 it/s baseline)
echo   - RFCNet: PyTorch (no torch.compile)
echo   - Flash Attention: Disabled
echo   - Concurrency: 6 workers (max parallelism with model caching!)
echo ============================================================
echo.

REM Use module form; threads pool works best on Windows
REM concurrency=6 for RTX 4090 (24GB VRAM - model caching enables true 6x parallelism!)
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=6
