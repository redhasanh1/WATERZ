@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
REM Minimal environment: only TensorRT-related config
echo Starting Celery worker (TensorRT-optimized)...

REM Activate Visual Studio 2022 C++ environment (required for torch.compile)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo ✅ Visual Studio C++ environment activated
)

REM Ensure TensorRT DLLs and tools are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM Force TensorRT-only RAFT (no PyTorch fallback)
set FORCE_TRT_RAFT=1

REM Enable Torch-TensorRT for RFCNet (Blackwell-compatible hybrid compilation)
set RFCNET_TORCHTRT=1

REM Disable Flash Attention for testing
set ENABLE_FLASH_ATTENTION=0

REM Use NeuFlow v2 exclusively (no RAFT fallback)
set USE_NEUFLOW=1

REM Use module form; threads pool works best on Windows
REM concurrency=2 is a safe default for RTX 5070
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=2 
