@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo Starting LOCAL TensorRT test (same config as Celery worker)...
echo.
echo ============================================================
echo LOCAL TENSORRT TEST: videostotrain\first.mp4
echo SAME CONFIG AS CELERY WORKER (no upload/download overhead)
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

REM ⚡ SEGMENT_WORKERS: Number of parallel ProPainter segment workers
set SEGMENT_WORKERS=4

echo.
echo ============================================================
echo OPTIMIZED CONFIG (RTX 4090):
echo   - YOLO: TensorRT batch 64 (FASTEST! 748 fps benchmark)
echo   - Optical Flow: NeuFlow v2 ONNX (10-70x faster than RAFT!)
echo   - RFCNet: PyTorch (no torch.compile)
echo   - Flash Attention: ENABLED (3-5x transformer speedup!)
echo   - Input: videostotrain\first.mp4
echo   - Output: results\first_no_watermark.mp4
echo ============================================================
echo.

REM Run local test with same environment as Celery
"%PYTHON_PATH%" test_local_propainter_trt.py

pause
