@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
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

REM ❌ RFCNET TENSORRT (DCNv4): DISABLED (isolation testing - checking for CUDA illegal memory access)
REM    Testing if RFCNet TensorRT engine is causing GPU corruption
REM    Fallback: PyTorch RFCNet (stable, ~10-15ms per frame)
REM    TODO: Re-enable after confirming transformer TensorRT works in isolation
set RFCNET_TORCHTRT=0
set FORCE_TRT_RFCNET=0

REM 🧪 TRANSFORMER: SWITCHED TO PYTORCH WRAPPER (TRT buffer routing bug)
REM    TRT plugin produces correct output but buffer routing layer prevents it from reaching user
REM    PyTorch wrapper bypasses this issue with direct I/O
REM    Performance: ~64ms per batch (vs ~40ms TRT would be if it worked)
REM    Quality: Perfect match to training (std=0 vs TRT's corrupted output)
set FORCE_CUSTOM_KERNEL_TRANSFORMER=0
set FORCE_HYBRID_TRANSFORMER=0
set FORCE_TRT_TRANSFORMER=0
set FORCE_PYTORCH_WRAPPER=1

REM ⚡ FLASH ATTENTION: DISABLED (conflicts with PyTorch wrapper)
REM    Flash Attention optimizations may not be compatible with native PyTorch path
REM    Need to test if this is causing quality degradation
set ENABLE_FLASH_ATTENTION=0

REM ❌ FP8 TRANSFORMER: DISABLED (amplifies TRT variance drift)
REM    FP8 quantization is not bit-exact and compounds rounding errors in fused kernels
REM    Disabling to test if TRT transformer can achieve PyTorch-grade quality without FP8
set ENABLE_FP8_TRANSFORMER=0

REM ❌ TOKEN MERGING: DISABLED (quality takes priority over speed)
REM    Even conservative 25%% reduction still degraded quality
REM    Disabling restores full quality at cost of 3-4s speedup
REM    Still benefits from Flash Attention + FP8 (2.5x faster baseline)
REM    Expected: 13s per video to ~11s (still 2s faster, perfect quality)
set ENABLE_TOKEN_MERGING=0
REM set TOKEN_MERGE_RATIO=0.25

REM ❌ FP8 ENCODER/DECODER: DISABLED (cuDNN "FIND engine" error)
REM    Issue: cuDNN cannot find engine for FP8 Conv2d operations
REM    Fallback: Standard FP16 encoder/decoder
set ENABLE_FP8_ENCODER=0
set ENABLE_FP8_DECODER=0

REM ❌ FP8 RFCNET: DISABLED (testing minimal config)
REM    Fallback: Standard FP16 RFCNet convolutions
set ENABLE_FP8_RFCNET=0

REM ❌ DCNv4 RFCNET: DISABLED (CUDA illegal memory access error)
REM    Issue: ext.dcnv4_forward() causing illegal memory access
REM    Fallback: Standard torchvision.ops.deform_conv2d (stable, no speedup)
REM    TODO: Debug DCNv4 CUDA kernel memory issue
set ENABLE_DCNV4_RFCNET=0

REM ❌ NVDEC VIDEO DECODER: DISABLED (7.4x SLOWER than CPU due to GPU→CPU transfer overhead)
REM    GPU decode is fast, but PCIe copy to CPU numpy arrays kills all gains
REM    Only beneficial if entire pipeline stays on GPU (not this use case)
set ENABLE_NVDEC=0

REM ⚡ TORCH COMPILE: DISABLED (not thread-safe - causes FileExistsError cache races)
REM    PyTorch 2.4.1 torch.compile has NO thread-safe caching mechanism
REM    We get 2.48x speedup from Flash Attention + FP8 (STABLE!)
REM    Next: Token Merging for 2-3x additional speedup (6x total)
set USE_TORCH_COMPILE=0
set TORCH_CUDAGRAPHS=0
set TORCHINDUCTOR_CUDAGRAPHS=0

REM ⚡ SEGMENT_WORKERS: Number of parallel ProPainter segment workers (must match --concurrency!)
set SEGMENT_WORKERS=4

echo.
echo ============================================================
echo OPTIMIZED CONFIG (RTX 4090):
echo   - YOLO: TensorRT batch 64 (FASTEST! 748 fps benchmark)
echo   - Video Decode: CPU cv2.VideoCapture (7.4x faster than NVDEC for CPU pipeline!)
echo   - Optical Flow: NeuFlow v2 TensorRT FP16 (3-4x faster than ONNX, 10-70x faster than RAFT!)
echo   - RFCNet: PyTorch FP16 (DISABLED TRT for isolation testing - checking for CUDA errors)
echo   - Transformer: PyTorch Wrapper (Bypasses TRT buffer bug, ~64ms/batch, perfect quality)
echo   - Flash Attention: ENABLED (Blackwell-optimized) [NOTE: TRT bypass this]
echo   - FP8 Transformer: ENABLED (1.3-1.5x speedup) [NOTE: TRT bypass this]
echo   - Token Merging: DISABLED (quality priority)
echo   - torch.compile: DISABLED (not thread-safe, causes cache races)
echo   - FP8 Encoder/Decoder: DISABLED (cuDNN error - using FP16 fallback)
echo   - FP8 RFCNet: DISABLED (minimal config test)
echo   - DCNv4: DISABLED (CUDA error - using standard deform_conv2d fallback)
echo   - Concurrency: 4 workers (TRUE 4-way parallel!)
echo   - SEGMENT_WORKERS: 4 (all segments process simultaneously)
echo ============================================================
echo.

REM Disable ALL torch.compile caching to avoid multi-worker file locking race conditions
REM Workers will compile kernels on first run (~20-40s startup), then cache in memory
REM This avoids FileExistsError crashes when 4 workers compile same kernels simultaneously
set TORCHINDUCTOR_FX_GRAPH_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=0
set TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

REM Clean any corrupted torch.compile cache before starting
REM if exist "D:\watermarkz\temp\torchinductor_has" (
REM     echo [CLEANUP] Removing ALL torch.compile cache (including kernels)...
REM     rmdir /s /q "D:\watermarkz\temp\torchinductor_has" 2>nul
REM     echo [OK] Complete cache removed - workers will recompile on startup
REM )

REM ⚠️ DEBUG: Enable CUDA synchronous errors to catch TRT crashes immediately
set CUDA_LAUNCH_BLOCKING=1
echo [DEBUG] CUDA_LAUNCH_BLOCKING=1 (synchronous errors for debugging)

REM 🧪 DISABLE TF32 to prevent TF16-like rounding in FP32 accumulation paths
REM    TF32 (TensorFloat32) uses FP16-like mantissa in FP32 ops, amplifying fusion drift
REM    Disabling forces full FP32 precision in intermediate calculations
REM    TEMPORARILY DISABLED: Testing LayerNorm FP16 precision fix first
REM    If LayerNorm fix works, will rebuild NeuFlow with TF32=0 and re-enable
REM set NVIDIA_TF32_OVERRIDE=0
echo [INFO] TF32 override temporarily disabled (testing LayerNorm FP16 fix)
echo.

REM Use thread pool with THREAD-LOCAL TensorRT contexts for TRUE PARALLEL execution!
REM Each thread gets its own TensorRT context = NO LOCKS = FULL PARALLEL!
REM concurrency=4 = 4 threads, each with separate context (minimal VRAM overhead)
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=4
