@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM ✅ AUTO-LOAD REDIS_URL from redis_url.txt
if exist redis_url.txt (
    for /f "usebackq tokens=*" %%A in ("redis_url.txt") do set REDIS_URL=%%A
    echo [REDIS] Loaded from redis_url.txt: %REDIS_URL%
) else (
    echo [REDIS] Using default localhost (redis_url.txt not found)
    set REDIS_URL=redis://:watermarkz_secure_2024@localhost:6379/0
)

REM ✅ AUTO-LOAD TUNNEL_URL from web\tunnel_url.txt for Railway upload
if exist web\tunnel_url.txt (
    for /f "usebackq delims=" %%A in ("web\tunnel_url.txt") do (
        if not defined TUNNEL_URL set TUNNEL_URL=%%A
    )
    echo [UPLOAD] Loaded TUNNEL_URL from web\tunnel_url.txt: %TUNNEL_URL%
    echo [UPLOAD] ✅ Auto-upload to Railway ENABLED
) else (
    echo [UPLOAD] ⚠️  web\tunnel_url.txt not found - Railway upload disabled
    echo [UPLOAD] 💡 Create web\tunnel_url.txt with your Railway URL to enable auto-upload
)
echo.

REM Enable auto-upload by default (set to 0 to disable)
if not defined UPLOAD_RESULT_BACK set UPLOAD_RESULT_BACK=1

REM Debug: Show all upload-related environment variables
echo [DEBUG] TUNNEL_URL = %TUNNEL_URL%
echo [DEBUG] API_BASE_URL = %API_BASE_URL%
echo [DEBUG] UPLOAD_RESULT_BACK = %UPLOAD_RESULT_BACK%
echo.

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
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM Force TensorRT-only RAFT (no PyTorch fallback)
set FORCE_TRT_RAFT=1

REM Disable RFCNet torch.compile (requires Triton which isn't available on Windows)
set RFCNET_TORCHTRT=0

REM Disable Flash Attention for testing
set ENABLE_FLASH_ATTENTION=0

REM ENABLE NeuFlow v2 for testing (FAIL-FAST MODE - no fallbacks!)
set USE_NEUFLOW=1

REM Use module form; threads pool works best on Windows
REM concurrency=4 recommended for RTX 4090 (24GB VRAM - can handle 4-5 parallel segments)
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=4 
