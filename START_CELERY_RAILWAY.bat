@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ========================================
echo   CELERY WORKER - RAILWAY REDIS MODE
echo ========================================
echo.
echo This connects your local GPU workers to Railway Redis
echo Make sure Railway Redis URL is set in redis_url.txt
echo.

REM Load Railway Redis URL from redis_url.txt
if exist redis_url.txt (
    for /f "usebackq tokens=*" %%A in ("redis_url.txt") do set REDIS_URL=%%A
    echo [REDIS] Loaded Railway URL: %REDIS_URL%
) else (
    echo [ERROR] redis_url.txt not found!
    echo Please create redis_url.txt with your Railway Redis URL
    echo Example: redis://default:password@railway-redis.railway.app:6379
    pause
    exit /b 1
)
echo.

REM Activate Visual Studio 2022 C++ environment (required for torch.compile)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo [MSVC] Visual Studio C++ environment activated
)

REM Ensure TensorRT DLLs and tools are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM Force TensorRT-only RAFT (no PyTorch fallback)
set FORCE_TRT_RAFT=1

REM Disable RFCNet torch.compile (requires Triton which isn't available on Windows)
set RFCNET_TORCHTRT=0

REM Disable Flash Attention for stability
set ENABLE_FLASH_ATTENTION=0

REM Enable NeuFlow v2 (TensorRT optimized)
set USE_NEUFLOW=1

echo.
echo Starting Celery workers with TensorRT optimizations...
echo Concurrency: 4 workers (optimized for RTX 4090 24GB VRAM)
echo.

REM Use module form; threads pool works best on Windows
REM concurrency=4 recommended for RTX 4090 (24GB VRAM - can handle 4-5 parallel segments)
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=4
