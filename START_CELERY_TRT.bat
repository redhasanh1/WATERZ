@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo Starting Celery worker (TensorRT FastFlowNet - optimized optical flow)...
echo.
echo ============================================================
echo TENSORRT MODE: FastFlowNet TensorRT FP16 (thread-safe, optimized graph)
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
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM Use NeuFlow v2 ONNX (10-70x faster than RAFT!)
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=1

REM Disable RFCNet torch.compile (requires Triton which isn't available on Windows)
set RFCNET_TORCHTRT=0

REM ⚡ FLASH ATTENTION: 3-5x speedup on transformer attention operations (FREE!)
set ENABLE_FLASH_ATTENTION=1

REM ⚡ SEGMENT_WORKERS: Number of parallel ProPainter segment workers (must match --concurrency!)
set SEGMENT_WORKERS=4

echo.
echo ============================================================
echo OPTIMIZED CONFIG (RTX 4090):
echo   - YOLO: TensorRT batch 64 (FASTEST! 748 fps benchmark)
echo   - Optical Flow: NeuFlow v2 ONNX (10-70x faster than RAFT!)
echo   - RFCNet: PyTorch (no torch.compile)
echo   - Flash Attention: ENABLED (3-5x transformer speedup!)
echo   - Concurrency: 4 workers (TRUE 4-way parallel!)
echo   - SEGMENT_WORKERS: 4 (all segments process simultaneously)
echo ============================================================
echo.

REM Use thread pool with THREAD-LOCAL TensorRT contexts for TRUE PARALLEL execution!
REM Each thread gets its own TensorRT context = NO LOCKS = FULL PARALLEL!
REM concurrency=4 = 4 threads, each with separate context (minimal VRAM overhead)
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=4
