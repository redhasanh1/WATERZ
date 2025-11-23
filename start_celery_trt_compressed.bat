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

REM Enable auto-upload by default (set to 0 to disable)
if not defined UPLOAD_RESULT_BACK set UPLOAD_RESULT_BACK=1

REM Debug: Show all upload-related environment variables
echo [DEBUG] TUNNEL_URL = %TUNNEL_URL%
echo [DEBUG] API_BASE_URL = %API_BASE_URL%
echo [DEBUG] UPLOAD_RESULT_BACK = %UPLOAD_RESULT_BACK%
echo.

echo Starting Celery worker (COMPRESSED MODE - Persistent Worker Pool!)...
echo.
echo ============================================================
echo 🚀 REVOLUTIONARY: PERSISTENT WORKER POOL MODE
echo ============================================================
echo.
echo This mode keeps models loaded in GPU memory between tasks:
echo   - First task:  20s (model loading + processing)
echo   - Next tasks:  8-10s (just processing, no loading!)
echo   - Speedup:     55%% faster for subsequent tasks
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

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM 🚀 REVOLUTIONARY: PERSISTENT WORKER POOL
REM    Models load ONCE per worker and stay in memory across all tasks
REM    Each worker processes segments in parallel with pre-loaded models
set USE_PERSISTENT_WORKERS=1
set MAX_PARALLEL_STREAMS=4
set PARALLEL_MODE=multiprocessing

REM ❌ TensorRT NeuFlow: DISABLED (using PyTorch + .TS files instead!)
REM    We want 4 PyTorch workers with .ts file loading
REM    NO TENSORRT - just pure PyTorch with pre-compiled .ts
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=0

REM ⚡ RFCNET: PyTorch mode (parallel-safe)
set RFCNET_TORCHTRT=0
set FORCE_TRT_RFCNET=0

REM ⚡ TRANSFORMER: Manual attention mode (fastest for sparse transformers)
set FORCE_TRT_TRANSFORMER=0

REM ❌ SAGEATTENTION: DISABLED (incompatible with ProPainter's sparse transformer!)
set ENABLE_SAGE_ATTENTION=0
set SAGE_CUDA_ARCH=89

REM ⚡ FLASH ATTENTION: DISABLED (manual attention is faster for sparse transformers!)
set ENABLE_FLASH_ATTENTION=0

REM ⚡ FP8 TRANSFORMER: 1.3-1.5x speedup on Linear layers (RTX 4090 Ada Lovelace 4th Gen Tensor Cores!)
set ENABLE_FP8_TRANSFORMER=1

REM ❌ TOKEN MERGING: DISABLED (quality takes priority over speed)
set ENABLE_TOKEN_MERGING=0

REM ⚡ FP8 ENCODER/DECODER: 1.3-1.5x speedup on Conv2d layers (11-16% pipeline speedup!)
set ENABLE_FP8_ENCODER=1
set ENABLE_FP8_DECODER=1

REM ⚡ FP8 RFCNET: 1.3-1.5x speedup on Conv2d/Conv3d layers (4-7% pipeline speedup!)
set ENABLE_FP8_RFCNET=1

REM ⚡ DCNv4 RFCNET: 3x speedup on deformable convolution (12-18% pipeline speedup!)
set ENABLE_DCNV4_RFCNET=1

REM ❌ NVDEC VIDEO DECODER: DISABLED (7.4x SLOWER than CPU due to GPU→CPU transfer overhead)
set ENABLE_NVDEC=0

REM ⚡ TORCH COMPILE: ENABLED (proven fast on Windows: 26-54ms/frame!)
REM    Works on Windows without Triton (inductor backend is sufficient)
REM    Tested: RUN_SAM2_LOCAL.py achieved 26-54ms/frame on Windows
set USE_TORCH_COMPILE_RAFT=1
set USE_TORCH_COMPILE_YOLO=1
set USE_TORCH_COMPILE_TRANSFORMER=0
set ENABLE_TRITON_KERNELS=1
set TORCH_CUDAGRAPHS=0
set TORCHINDUCTOR_CUDAGRAPHS=0

REM Clean worker output for production
set WORKER_VERBOSE=0

REM ⚡ SAM2 TRACKING: DISABLED (using YOLO-only for speed)
set USE_SAM2_TRACKING=0

echo.
echo ============================================================
echo PERSISTENT WORKER POOL CONFIG:
echo   - YOLO: TensorRT batch 64 (748 fps benchmark)
echo   - SAM2-Tiny: DISABLED (YOLO-only mode)
echo   - Video Decode: CPU cv2.VideoCapture (7.4x faster than NVDEC!)
echo   - Optical Flow: PyTorch RAFT + torch.compile (26-54ms/frame)
echo   - RFCNet: PyTorch + FP8 + DCNv4 (parallel-safe flow completion!)
echo   - Transformer: Manual attention (0.99-1.02s feature propagation!)
echo   - FP8 Transformer: ENABLED (RTX 4090: 1.3-1.5x speedup!)
echo   - FP8 Encoder/Decoder: ENABLED (11-16%% pipeline gain!)
echo   - FP8 RFCNet: ENABLED (4-7%% pipeline gain!)
echo   - DCNv4: ENABLED (3x faster deformable convolution!)
echo   - 🚀 PERSISTENT WORKERS: Models stay loaded across tasks!
echo   - 🚀 PARALLEL SEGMENTS: 4 workers process segments simultaneously!
echo   - Concurrency: 1 worker (dedicated, persistent model cache)
echo ============================================================
echo.
echo 🔥 PERFORMANCE EXPECTATIONS:
echo   Task 1:  ~20s (model loading + processing)
echo   Task 2+: ~10s (no loading, 55%% faster!)
echo   GPU:     Models stay in VRAM between tasks
echo   Cache:   torch.compile kernels cached for instant warmup
echo ============================================================
echo.

REM Enable per-worker torch.compile cache isolation
set TORCHINDUCTOR_FX_GRAPH_CACHE=1
set TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
set TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

REM 🚀 PERSISTENT MODE: Single worker with models loaded persistently
REM    Worker stays alive, models stay in GPU memory
REM    Each task reuses the same loaded models (55%% faster after first task!)
REM    Use --pool=solo to ensure single worker process
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=solo --concurrency=1
