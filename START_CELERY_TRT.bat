@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM ✅ AUTO-LOAD REDIS_URL from redis_url.txt
set REDIS_CONFIG_FILE=%~dp0redis_url.txt
echo [DEBUG] Looking for redis_url.txt at: %REDIS_CONFIG_FILE%

if exist "%REDIS_CONFIG_FILE%" (
    echo [DEBUG] File exists, reading...
    REM Read first line of redis_url.txt directly
    set /p REDIS_URL=<"%REDIS_CONFIG_FILE%"
    echo [REDIS] ✅ Loaded from redis_url.txt
    echo [REDIS] Value: %REDIS_URL%
) else (
    echo [REDIS] ❌ redis_url.txt not found at: %REDIS_CONFIG_FILE%
    echo [REDIS] Using default localhost fallback
    set REDIS_URL=redis://:watermarkz_secure_2024@localhost:6379/0
)

echo [DEBUG] Final REDIS_URL = %REDIS_URL%
echo.

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

echo Starting Celery worker (TensorRT NeuFlow v2 FP16 - 3-4x faster than ONNX!)...
echo.
echo ============================================================
echo TENSORRT MODE: NeuFlow v2 TensorRT FP16 (4 execution contexts, pure GPU)
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

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM Use NeuFlow v2 ONNX (10-70x faster than RAFT!)
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=1

REM ⚡ RFCNET TENSORRT (DCNv4): 1.6-2.3x speedup on flow completion (saves ~0.8-1.0s per video!)
REM    Uses pre-built FP16 TensorRT engine with DCNv4 plugin (target: 7-10ms @ 640x480)
REM    Proven performance: 9.3ms/frame achieved (consistent after ~500ms first-segment warmup)
REM    NO FALLBACK: TensorRT-only mode (will fail if engine missing)
set RFCNET_TORCHTRT=0
set FORCE_TRT_RFCNET=1

REM ⚡ TRANSFORMER TENSORRT: 5-10x speedup on feature propagation (2.39s → 0.24-0.48s per segment!)
REM    Uses pre-built FP16 TensorRT engine for Sparse Temporal Transformer (8 layers, 4 heads)
REM    Current: 2.39s per segment (45-65%% of pipeline) with PyTorch FP8 + Flash Attention
REM    Target:  0.24-0.48s per segment (5-10x faster!) - transformer no longer bottleneck
REM    NO FALLBACK: TensorRT-only mode (will fail if engine missing)
REM    Build engine with: BUILD_TRANSFORMER_ENGINE.bat
set FORCE_TRT_TRANSFORMER=0

REM ❌ SAGEATTENTION: DISABLED (incompatible with ProPainter's sparse transformer!)
REM    ProPainter uses grouped-query attention with window-based sparsity
REM    SageAttention only supports standard dense attention (BERT/GPT/ViT)
REM    Manual attention is FASTER: 0.99-1.02s vs 4.83-5.22s with SageAttention fallback
set ENABLE_SAGE_ATTENTION=0
set SAGE_CUDA_ARCH=89

REM ⚡ FLASH ATTENTION: DISABLED (manual attention is faster for sparse transformers!)
REM    ProPainter's optimized manual attention: 0.99-1.02s feature propagation
REM    Flash Attention/SDPA fallback: 4.83-5.22s (5x slower!)
set ENABLE_FLASH_ATTENTION=0

REM ⚡ FP8 TRANSFORMER: 1.3-1.5x speedup on Linear layers (RTX 4090 Ada Lovelace 4th Gen Tensor Cores!)
REM    Combined with DCNv4 + Flash Attention = 5-10x faster transformers!
set ENABLE_FP8_TRANSFORMER=1

REM ❌ TOKEN MERGING: DISABLED (quality takes priority over speed)
REM    Even conservative 25%% reduction still degraded quality
REM    Disabling restores full quality at cost of 3-4s speedup
REM    Still benefits from Flash Attention + FP8 (2.5x faster baseline)
REM    Expected: 13s per video to ~11s (still 2s faster, perfect quality)
set ENABLE_TOKEN_MERGING=0
REM set TOKEN_MERGE_RATIO=0.25

REM ⚡ FP8 ENCODER/DECODER: 1.3-1.5x speedup on Conv2d layers (11-16% pipeline speedup!)
REM    Uses FP8Conv2d for all encoder/decoder convolutions (Ada 4th Gen Tensor Cores)
set ENABLE_FP8_ENCODER=1
set ENABLE_FP8_DECODER=1

REM ⚡ FP8 RFCNET: 1.3-1.5x speedup on Conv2d/Conv3d layers (4-7% pipeline speedup!)
REM    Uses FP8Conv2d/FP8Conv3d for flow completion network (Ada 4th Gen Tensor Cores)
set ENABLE_FP8_RFCNET=1

REM ⚡ DCNv4 RFCNET: 3x speedup on deformable convolution (12-18% pipeline speedup!)
REM    Uses DCNv4 module with optimized CUDA kernels (replaces torchvision deform_conv2d)
REM    Phase 1A: PyTorch validation (3x speedup expected)
REM    Phase 2-4: TensorRT plugin (6.5-9x total speedup when combined with FP8)
set ENABLE_DCNV4_RFCNET=1

REM ❌ NVDEC VIDEO DECODER: DISABLED (7.4x SLOWER than CPU due to GPU→CPU transfer overhead)
REM    GPU decode is fast, but PCIe copy to CPU numpy arrays kills all gains
REM    Only beneficial if entire pipeline stays on GPU (not this use case)
set ENABLE_NVDEC=0

REM ⚡ TORCH COMPILE: DISABLED for transformer (Windows FileExistsError issue)
REM    Keep RAFT compilation enabled (specific flag, no FileExistsError)
REM    Disabling transformer compilation prevents random pixel artifacts
set USE_TORCH_COMPILE=0
set USE_TORCH_COMPILE_RAFT=1
set TORCH_CUDAGRAPHS=0
set TORCHINDUCTOR_CUDAGRAPHS=0

REM ⚡ SEGMENT_WORKERS: Number of parallel ProPainter segment workers (must match --concurrency!)
REM    Parallel mode: 4 workers (per-PID cache isolation prevents conflicts)
set SEGMENT_WORKERS=4

REM ⚡ SAM2 TRACKING: Use SAM2-Tiny for temporal mask tracking (BEST QUALITY!)
REM    YOLO detects bbox on first frame → SAM2 tracks with temporal consistency
REM    44ms/frame, no flickering, perfect for moving watermarks!
REM TEMPORARILY DISABLED - using YOLO-only for speed testing
set USE_SAM2_TRACKING=0

echo.
echo ============================================================
echo OPTIMIZED CONFIG (RTX 4090):
echo   - YOLO: TensorRT batch 64 (FASTEST! 748 fps benchmark)
echo   - SAM2-Tiny: DISABLED (YOLO-only mode for speed)
echo   - Video Decode: CPU cv2.VideoCapture (7.4x faster than NVDEC for CPU pipeline!)
echo   - Optical Flow: NeuFlow v2 TensorRT FP16 (3-4x faster than ONNX, 10-70x faster than RAFT!)
echo   - RFCNet: TensorRT FP16 + DCNv4 plugin (1.6-2.3x faster flow completion!)
echo   - Transformer: Manual attention (FASTEST! 0.99-1.02s feature propagation!)
echo   - SageAttention: DISABLED (incompatible with sparse transformers!)
echo   - Flash Attention: DISABLED (manual attention is 5x faster!)
echo   - FP8 Transformer: DISABLED (torch.compile optimizes instead!)
echo   - Token Merging: DISABLED (quality priority)
echo   - torch.compile: ENABLED (sequential = safe, 1.5-2x RAFT speedup!)
echo   - FP8 Encoder/Decoder: DISABLED (precision artifacts fixed!)
echo   - FP8 RFCNet: ENABLED (1.3-1.5x speedup, 4-7%% pipeline gain!)
echo   - DCNv4: ENABLED (3x faster deformable convolution!)
echo   - Concurrency: solo (SINGLE WORKER - last segment triggers finalize!)
echo   - SEGMENT_WORKERS: 4 (segments processed sequentially)
echo   - torch.compile: ENABLED (1.5-2x speedup on RAFT!)
echo ============================================================
echo.

REM ✅ ENABLE torch.compile caching to match local script behavior (stable + fast!)
REM Per-PID cache isolation (cache_config.py) prevents multi-worker conflicts
REM Matches RUN_SAM2_LOCAL.BAT settings for identical quality + speed
set TORCHINDUCTOR_FX_GRAPH_CACHE=1
set TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
set TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

REM Clean any corrupted torch.compile cache before starting
REM if exist "D:\watermarkz\temp\torchinductor_has" (
REM     echo [CLEANUP] Removing ALL torch.compile cache (including kernels)...
REM     rmdir /s /q "D:\watermarkz\temp\torchinductor_has" 2>nul
REM     echo [OK] Complete cache removed - workers will recompile on startup
REM )

REM Sequential mode: 1 worker for torch.compile safety
REM concurrency=1 = 1 thread, safe for torch.compile (no cache conflicts)

echo.
echo ============================================================
echo STARTING CELERY WORKER...
echo ============================================================
echo [DEBUG] Connecting to Redis broker: %REDIS_URL%
echo.
echo [TEST] Testing Redis connectivity...
echo [DEBUG] Batch file REDIS_URL: %REDIS_URL%

REM Test Redis connection using Python - Pass URL as command line arg instead of env var
echo | set /p="[TEST] Attempting connection... "
"%PYTHON_PATH%" -c "import redis, sys; url = '%REDIS_URL%'; r = redis.from_url(url, socket_connect_timeout=3, socket_timeout=3); r.ping(); print('[OK] Connection successful!'); sys.exit(0)"
if errorlevel 1 (
    echo FAILED
    echo.
    echo [ERROR] ❌ Cannot connect to Redis server!
    echo [ERROR] Redis URL attempted: %REDIS_URL%
    echo.
    echo [FIX] Options:
    echo   1. Check redis_url.txt has correct Railway Redis URL
    echo   2. Verify Railway Redis service is running
    echo   3. Check network/firewall allows connection to Railway
    echo   4. For local dev: Start Redis locally with: redis-server
    echo.
    pause
    exit /b 1
)

echo [INFO] This may take 10-30s for model loading...
echo.

"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=solo
