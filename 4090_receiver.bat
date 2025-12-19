@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM ============================================================
REM 4090 RECEIVER - Windows ProPainter Worker
REM ============================================================
REM   - Downloads masks from B2 CDN
REM   - Detects segments
REM   - Runs ProPainter inpainting
REM   - Uploads result to B2
REM ============================================================

REM AUTO-LOAD REDIS_URL from redis_url.txt
if exist redis_url.txt (
    set /p REDIS_URL=<redis_url.txt
    echo [REDIS] Loaded from redis_url.txt
) else (
    echo [REDIS] Using default localhost (redis_url.txt not found)
    set REDIS_URL=redis://:watermarkz_secure_2024@localhost:6379/0
)

REM AUTO-LOAD TUNNEL_URL from tunnel_url.txt (root) or web\tunnel_url.txt (needed for video download)
if "%TUNNEL_URL%"=="" (
    if exist "tunnel_url.txt" (
        set /p TUNNEL_URL=<tunnel_url.txt
    ) else if exist "web\tunnel_url.txt" (
        set /p TUNNEL_URL=<web\tunnel_url.txt
    )
)
if not "%TUNNEL_URL%"=="" (
    set "API_BASE_URL=%TUNNEL_URL%"
    echo [TUNNEL] Using TUNNEL_URL: %TUNNEL_URL%
) else (
    echo [TUNNEL] No TUNNEL_URL set - local paths only
)
echo.

REM B2 credentials for mask download and result upload
set B2_KEY_ID=00539db5c1104b50000000003
set B2_APP_KEY=K005384b8lPoBT11wScxkZ2Gx0fszus
set B2_BUCKET=watermarkz
set B2_CDN_URL=https://markz.humblewoslayer.workers.dev
set B2_UPLOAD_ENABLED=1
echo [B2] Credentials loaded for %B2_BUCKET%
echo.

echo.
echo ============================================================
echo 4090 RECEIVER - ProPainter Worker
echo ============================================================
echo   - Downloads masks from B2 CDN
echo   - Detects segments (motion-based)
echo   - In-memory ProPainter (ZERO disk I/O)
echo   - NeuFlow TRT optical flow (10-70x faster)
echo   - Uploads result to B2 CDN
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

REM Ensure TensorRT DLLs and tools are on PATH (guarded)
if exist "%~dp0SETUP_TRT_ENV.bat" (
  call "%~dp0SETUP_TRT_ENV.bat"
) else (
  echo [WARN] SETUP_TRT_ENV.bat not found; ensure TensorRT is on PATH
)

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM  SAM2 INTERACTIVE MODE - NO YOLO DETECTION!
set YOLO_REQUIRE_TENSORRT=0
set USE_INTERACTIVE_SAM2=1

REM NeuFlow TRT ENABLED - all others OFF for testing
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=1

REM  RFCNET TENSORRT: ENABLED for best quality
set RFCNET_TORCHTRT=0
set FORCE_TRT_RFCNET=1

REM  TRANSFORMER TENSORRT: DISABLED for testing
set FORCE_TRT_TRANSFORMER=0

REM  SAGEATTENTION: DISABLED (incompatible with ProPainter's sparse transformer!)
set ENABLE_SAGE_ATTENTION=0
set SAGE_CUDA_ARCH=89

REM  FLASH ATTENTION: DISABLED (manual attention is faster!)
set ENABLE_FLASH_ATTENTION=0

REM  FP8 TRANSFORMER: 1.3-1.5x speedup on Linear layers
set ENABLE_FP8_TRANSFORMER=1

REM  TOKEN MERGING: DISABLED (quality takes priority)
set ENABLE_TOKEN_MERGING=0

REM  FP8 ENCODER/DECODER: 1.3-1.5x speedup
set ENABLE_FP8_ENCODER=1
set ENABLE_FP8_DECODER=1

REM  FP8 RFCNET: 1.3-1.5x speedup
set ENABLE_FP8_RFCNET=1

REM  DCNv4 RFCNET: ENABLED for best quality
set ENABLE_DCNV4_RFCNET=1

REM  NVDEC VIDEO DECODER: DISABLED
set ENABLE_NVDEC=0

REM  TORCH COMPILE: DISABLED (not thread-safe)
set USE_TORCH_COMPILE=0
set TORCH_CUDAGRAPHS=0
set TORCHINDUCTOR_CUDAGRAPHS=0

REM  SEGMENT_WORKERS: Number of parallel ProPainter segment workers (2 = share preloaded frames)
set SEGMENT_WORKERS=2

REM Prefer point prompt for SAM2 tracking (higher quality); set to bbox to force bbox mode
set SAM2_PROMPT_MODE=point

REM Stronger mask dilation for better coverage (1-12 recommended)
set SAM2_MASK_DILATION=4

REM Use strict full-video inpainting (no segment splitting) by default
set SAM2_USE_SEGMENTS=1
set SAM2_PARALLEL_SEGMENTS=1
set SAM2_SEGMENT_DETECTION_MODE=full

REM Motion-based segment detection (better for fast-moving objects like footballs)
REM Compares frame-to-frame movement instead of drift from average
set SEGMENT_USE_MOTION_DETECTION=1
set SEGMENT_MOTION_THRESHOLD=8
set SEGMENT_MIN_LEN_FULL=3
set SEGMENT_MERGE_GAP_FULL=10

REM Max frames per segment (prevents OOM on long videos - splits into chunks)
set MAX_SEGMENT_FRAMES=300

REM Max pixels (width*height) per segment's union bbox - prevents huge crops that slow processing
REM 400000 = ~630x630, keeps per-frame under 200ms. Lower = more segments = faster per segment
set MAX_SEGMENT_PIXELS=400000

REM Max pixels AFTER padding - triggers segment splitting AND downsampling if exceeded
REM 150k = ~387x387 max - very aggressive downsampling for safety
set MAX_CROP_PIXELS=150000

REM Background reference image for large watermark blending (leave empty to disable)
REM The background is warped using optical flow and blended with ProPainter output
set BACKGROUND_IMAGE_PATH=D:\watermarkz\videostotrain\firstphoto.jpg
REM Blend ratio: 0.0 = pure ProPainter, 0.5 = balanced, 1.0 = pure background
set BACKGROUND_ALPHA=0.5

REM Legacy average-based detection params (used when SEGMENT_USE_MOTION_DETECTION=0)
set SEGMENT_POS_TOLERANCE=50
set SEGMENT_MIN_LEN_10FPS=3
set SEGMENT_MERGE_GAP_10FPS=6

REM  SAM2 TRACKING: NOT USED IN INTERACTIVE MODE (masks provided by user)
set USE_SAM2_TRACKING=0

REM Windows-side FPS downsample for WSL speed (faster but less accurate for fast objects)
set SAM2_TRACK_DOWNSAMPLE=0
set SAM2_TRACK_FPS=15

echo.
echo ============================================================
echo ProPainter CONFIG (RTX 4090):
echo   - Queue: propainter
echo   - ProPainter: In-memory arrays (ZERO disk I/O!)
echo   - Optical Flow: NeuFlow v2 TensorRT FP16
echo   - RFCNet: TensorRT FP16 + DCNv4 plugin
echo   - FP8 Optimizations: ENABLED
echo ============================================================
echo.

REM Disable torch.compile caching
set TORCHINDUCTOR_FX_GRAPH_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

REM Use thread pool with 4 workers for TRUE parallel segment processing
set CELERY_POOL=threads
set CELERY_CONCURRENCY=4

echo.
echo ============================================================
echo UNIFIED WORKER - SAM2 + YOLO in ONE process
echo ============================================================
echo   - Queues: celery (YOLO chord), propainter (SAM2)
echo   - Single process = less VRAM, simpler management
echo ============================================================
echo.

REM UNIFIED WORKER: server_production2 imports server_production tasks
REM Handles both SAM2 interactive AND YOLO chord in ONE worker
"%PYTHON_PATH%" -u -m celery -A server_production2.celery worker -Q celery,propainter --loglevel=info --pool=threads --concurrency=4
