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
echo.

echo Starting Celery worker (SAM2 INTERACTIVE MODE - TensorRT optimized!)...
echo.
echo ============================================================
echo SAM2 INTERACTIVE MODE: User-click based object removal
echo ============================================================
echo.

REM Activate Visual Studio 2022 C++ environment (required for torch.compile)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo [OK] Visual Studio C++ environment activated
)

REM 🔥 SET CUDA PATH FOR NVDEC (PyNvVideoCodec requires CUDA DLLs!)
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
echo [CUDA] Set CUDA_PATH=%CUDA_PATH%

REM Ensure TensorRT DLLs and tools are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM ⚡ SAM2 INTERACTIVE MODE - NO YOLO DETECTION!
set YOLO_REQUIRE_TENSORRT=0
set USE_INTERACTIVE_SAM2=1

REM Use NeuFlow v2 ONNX (10-70x faster than RAFT!)
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=1

REM ⚡ RFCNET TENSORRT (DCNv4): 1.6-2.3x speedup on flow completion
set RFCNET_TORCHTRT=0
set FORCE_TRT_RFCNET=1

REM ⚡ TRANSFORMER TENSORRT: DISABLED (optional optimization)
set FORCE_TRT_TRANSFORMER=0

REM ❌ SAGEATTENTION: DISABLED (incompatible with ProPainter's sparse transformer!)
set ENABLE_SAGE_ATTENTION=0
set SAGE_CUDA_ARCH=89

REM ⚡ FLASH ATTENTION: DISABLED (manual attention is faster!)
set ENABLE_FLASH_ATTENTION=0

REM ⚡ FP8 TRANSFORMER: 1.3-1.5x speedup on Linear layers
set ENABLE_FP8_TRANSFORMER=1

REM ❌ TOKEN MERGING: DISABLED (quality takes priority)
set ENABLE_TOKEN_MERGING=0

REM ⚡ FP8 ENCODER/DECODER: 1.3-1.5x speedup
set ENABLE_FP8_ENCODER=1
set ENABLE_FP8_DECODER=1

REM ⚡ FP8 RFCNET: 1.3-1.5x speedup
set ENABLE_FP8_RFCNET=1

REM ⚡ DCNv4 RFCNET: 3x speedup on deformable convolution
set ENABLE_DCNV4_RFCNET=1

REM ❌ NVDEC VIDEO DECODER: DISABLED
set ENABLE_NVDEC=0

REM ⚡ TORCH COMPILE: DISABLED (not thread-safe)
set USE_TORCH_COMPILE=0
set TORCH_CUDAGRAPHS=0
set TORCHINDUCTOR_CUDAGRAPHS=0

REM ⚡ SEGMENT_WORKERS: Number of parallel ProPainter segment workers
set SEGMENT_WORKERS=4

REM ⚡ SAM2 TRACKING: NOT USED IN INTERACTIVE MODE (masks provided by user)
set USE_SAM2_TRACKING=0

echo.
echo ============================================================
echo SAM2 INTERACTIVE CONFIG (RTX 4090):
echo   - Detection: SAM2 INTERACTIVE (user clicks, no YOLO!)
echo   - Optical Flow: NeuFlow v2 TensorRT FP16
echo   - RFCNet: TensorRT FP16 + DCNv4 plugin
echo   - Transformer: Manual attention (FASTEST!)
echo   - FP8 Optimizations: ENABLED
echo   - DCNv4: ENABLED (3x faster!)
echo   - Concurrency: 4 workers (TRUE parallel!)
echo ============================================================
echo.

REM Disable torch.compile caching
set TORCHINDUCTOR_FX_GRAPH_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

REM Purge old tasks from Redis queue before starting
echo.
echo [CLEANUP] Purging old tasks from Redis queue...
"%PYTHON_PATH%" -c "from server_production import celery; celery.control.purge(); print('[OK] Redis queue cleared')"
echo.

REM Use thread pool with 1 worker (sequential to prevent temp folder collisions)
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=1
