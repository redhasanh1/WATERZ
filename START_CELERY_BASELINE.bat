@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo Starting Celery worker (PyTorch BASELINE - no optimizations)...
echo.
echo ============================================================
echo BASELINE MODE: Pure PyTorch RAFT (reference for comparisons)
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
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
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
echo BASELINE CONFIG:
echo   - YOLO: TensorRT (600-1000 fps)
echo   - RAFT: PyTorch FP16 autocast (~10.5 it/s baseline)
echo   - RFCNet: PyTorch (no torch.compile)
echo   - Flash Attention: Disabled
echo ============================================================
echo.

REM Use module form; threads pool works best on Windows
REM concurrency=2 is a safe default for RTX 5070
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=2
