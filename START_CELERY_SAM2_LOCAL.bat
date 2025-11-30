@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo SAM2 WATERMARK REMOVAL - CELERY WORKER
echo (Isolated from YOLO production)
echo ============================================================
echo.

REM ✅ AUTO-LOAD REDIS_URL from redis_url.txt
if exist redis_url.txt (
    for /f "usebackq tokens=*" %%A in ("redis_url.txt") do set REDIS_URL=%%A
    echo [REDIS] Loaded from redis_url.txt: %REDIS_URL%
) else (
    echo [REDIS] Using default localhost
    set REDIS_URL=redis://:watermarkz_secure_2024@localhost:6379/0
)

echo.
echo ============================================================
echo SAM2 INTERACTIVE MODE ENABLED
echo ============================================================
echo.

REM 🔥 ENABLE SAM2 INTERACTIVE MODE
set USE_INTERACTIVE_SAM2=1

REM 🔥 SEGMENT WORKERS (must match --concurrency)
set SEGMENT_WORKERS=4

REM 🔥 CUDA PATH FOR NVENC
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

REM Ensure TensorRT DLLs are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM ============================================================
REM OPTICAL FLOW OPTIMIZATION
REM ============================================================
REM Use NeuFlow v2 TensorRT FP16 (10-70x faster than RAFT!)
set USE_NEUFLOW=1
set FORCE_TRT_RAFT=0

echo [OPTICAL FLOW] NeuFlow v2 TensorRT FP16 (10-70x faster than RAFT!)
echo.

REM ============================================================
REM PROPAINTER OPTIMIZATIONS
REM ============================================================

REM ⚡ RFCNET TENSORRT (DCNv4): 1.6-2.3x speedup
set FORCE_TRT_RFCNET=1
set ENABLE_DCNV4_RFCNET=1

REM ⚡ FP8 OPTIMIZATIONS (RTX 4090 Ada Lovelace)
set ENABLE_FP8_TRANSFORMER=1
set ENABLE_FP8_ENCODER=1
set ENABLE_FP8_DECODER=1
set ENABLE_FP8_RFCNET=1

REM ❌ DISABLED OPTIMIZATIONS
REM SageAttention incompatible with sparse transformers
set ENABLE_SAGE_ATTENTION=0

REM Manual attention is faster for ProPainter
set ENABLE_FLASH_ATTENTION=0

REM Not thread-safe
set USE_TORCH_COMPILE=0

REM Quality takes priority
set ENABLE_TOKEN_MERGING=0

REM Slower than CPU for this pipeline
set ENABLE_NVDEC=0

REM Disable YOLO (not needed for SAM2 mode)
set YOLO_REQUIRE_TENSORRT=0
set USE_SAM2_TRACKING=0

echo [PROPAINTER] Optimizations:
echo   - RFCNet: TensorRT FP16 + DCNv4 (1.6-2.3x speedup)
echo   - FP8 Transformer: ENABLED (1.3-1.5x speedup)
echo   - FP8 Encoder/Decoder: ENABLED (1.3-1.5x speedup)
echo   - FP8 RFCNet: ENABLED (1.3-1.5x speedup)
echo   - Manual Attention: ENABLED (fastest for sparse transformers)
echo   - SageAttention: DISABLED (incompatible)
echo   - Flash Attention: DISABLED (manual is faster)
echo   - torch.compile: DISABLED (not thread-safe)
echo.

REM ============================================================
REM SAM2 CONFIGURATION
REM ============================================================
REM Allow 50px watermark movement between segments
set SAM2_POSITION_TOLERANCE=50

REM Minimum 3 frames per segment
set SAM2_MIN_SEGMENT_LENGTH=3

REM Cap at 80 segments max
set SAM2_MAX_SEGMENTS=80

REM <10px movement = stationary (use ULTRA-FAST params)
set SAM2_STATIONARY_THRESHOLD=10

echo [SAM2] Configuration:
echo   - Position tolerance: 50px
echo   - Min segment length: 3 frames
echo   - Max segments: 80
echo   - Stationary threshold: 10px
echo.

REM ============================================================
REM DISABLE TORCH.COMPILE CACHING (not thread-safe)
REM ============================================================
set TORCHINDUCTOR_FX_GRAPH_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

REM Use explicit Python path
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

echo ============================================================
echo STARTING CELERY WORKER FOR SAM2
echo ============================================================
echo Configuration:
echo   - Pool: threads
echo   - Concurrency: 4 workers
echo   - Segment Workers: 4
echo   - Server: server_production2.py
echo ============================================================
echo.

REM Start Celery worker pointing to server_production2
REM -Q sam2 ensures this worker ONLY handles SAM2 tasks (ignores prepare_video etc)
"%PYTHON_PATH%" -m celery -A server_production2.celery worker -Q sam2 --loglevel=info --pool=threads --concurrency=4

pause
