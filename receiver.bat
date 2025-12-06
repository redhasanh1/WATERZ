@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM ============================================================
REM RECEIVER - Windows ProPainter Worker
REM ============================================================
REM Downloads masks from B2 CDN → ProPainter inpainting → upload result
REM SEGMENT_WORKERS is set dynamically by task metadata:
REM   - SAM2 mode: 2 workers
REM   - YOLO mode: 4 workers
REM ============================================================

echo.
echo ============================================================
echo RECEIVER - Windows ProPainter Worker
echo ============================================================
echo   - Downloads masks from B2 CDN
echo   - Runs ProPainter inpainting
echo   - Uploads result to B2 CDN
echo ============================================================
echo.

REM Load Redis URL from file or use default
if exist redis_url.txt (
    for /f "usebackq tokens=*" %%A in ("redis_url.txt") do set REDIS_URL=%%A
    echo [REDIS] Loaded from redis_url.txt
) else (
    set REDIS_URL=redis://:watermarkz_secure_2024@localhost:6379/0
    echo [REDIS] Using default localhost
)

REM Load tunnel URL for video downloads
if exist "tunnel_url.txt" (
    for /f "usebackq delims=" %%i in ("tunnel_url.txt") do set "TUNNEL_URL=%%i"
    set "API_BASE_URL=%TUNNEL_URL%"
    echo [TUNNEL] Using: %TUNNEL_URL%
) else if exist "web\tunnel_url.txt" (
    for /f "usebackq delims=" %%i in ("web\tunnel_url.txt") do set "TUNNEL_URL=%%i"
    set "API_BASE_URL=%TUNNEL_URL%"
    echo [TUNNEL] Using: %TUNNEL_URL%
) else (
    echo [TUNNEL] No tunnel URL set - local paths only
)
echo.

REM Activate Visual Studio 2022 C++ environment (for torch.compile)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo [OK] Visual Studio C++ environment activated
)

REM CUDA path for NVDEC/NVENC
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

REM TensorRT setup
if exist "%~dp0SETUP_TRT_ENV.bat" (
    call "%~dp0SETUP_TRT_ENV.bat"
) else (
    echo [WARN] SETUP_TRT_ENV.bat not found
)

REM Python path
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM ============================================================
REM ProPainter Optimizations
REM ============================================================

REM SAM2 mode - masks already provided
set YOLO_REQUIRE_TENSORRT=0
set USE_INTERACTIVE_SAM2=1

REM NeuFlow v2 for optical flow (10-70x faster than RAFT)
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=1

REM RFCNet TensorRT with DCNv4
set RFCNET_TORCHTRT=0
set FORCE_TRT_RFCNET=1

REM Transformer TensorRT (disabled - optional)
set FORCE_TRT_TRANSFORMER=0

REM SageAttention (disabled - incompatible with sparse transformer)
set ENABLE_SAGE_ATTENTION=0

REM Flash Attention (disabled - manual attention faster)
set ENABLE_FLASH_ATTENTION=0

REM FP8 optimizations
set ENABLE_FP8_TRANSFORMER=1
set ENABLE_FP8_ENCODER=1
set ENABLE_FP8_DECODER=1
set ENABLE_FP8_RFCNET=1

REM DCNv4 for RFCNet (3x speedup)
set ENABLE_DCNV4_RFCNET=1

REM Token merging (disabled - quality priority)
set ENABLE_TOKEN_MERGING=0

REM NVDEC video decoder (disabled)
set ENABLE_NVDEC=0

REM torch.compile (disabled - not thread-safe)
set USE_TORCH_COMPILE=0
set TORCH_CUDAGRAPHS=0
set TORCHINDUCTOR_CUDAGRAPHS=0

REM ============================================================
REM Segment Configuration
REM ============================================================
REM Default to 2 workers - task can override via metadata
set SEGMENT_WORKERS=2

REM SAM2 prompt mode
set SAM2_PROMPT_MODE=point
set SAM2_MASK_DILATION=4

REM Segment detection
set SAM2_USE_SEGMENTS=1
set SAM2_PARALLEL_SEGMENTS=1
set SAM2_SEGMENT_DETECTION_MODE=full

REM Motion-based detection
set SEGMENT_USE_MOTION_DETECTION=1
set SEGMENT_MOTION_THRESHOLD=8
set SEGMENT_MIN_LEN_FULL=3
set SEGMENT_MERGE_GAP_FULL=10

REM Limits
set MAX_SEGMENT_FRAMES=300
set MAX_SEGMENT_PIXELS=400000
set MAX_CROP_PIXELS=600000

REM SAM2 tracking (not used - masks provided by sender)
set USE_SAM2_TRACKING=0

REM Disable torch.compile caching
set TORCHINDUCTOR_FX_GRAPH_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

echo.
echo ============================================================
echo ProPainter CONFIG:
echo   - Queue: propainter
echo   - SEGMENT_WORKERS: %SEGMENT_WORKERS% (SAM2=2, YOLO=4)
echo   - Optical Flow: NeuFlow v2 TensorRT FP16
echo   - RFCNet: TensorRT FP16 + DCNv4
echo   - FP8 Optimizations: ENABLED
echo ============================================================
echo.

REM Start Celery worker
set CELERY_POOL=solo
set CELERY_CONCURRENCY=1
"%PYTHON_PATH%" -m celery -A server_production2.celery worker -Q propainter --loglevel=info --pool=solo
