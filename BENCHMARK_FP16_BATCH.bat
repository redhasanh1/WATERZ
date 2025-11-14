@echo off
cd /d "%~dp0"
echo ============================================================
echo FP16 TensorRT Batch Benchmark - RTX 4090 EXTREME SPEED!
echo ============================================================
echo.

REM Use Python 3.12
set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"

set ENGINE_PATH=runs\detect\new_sora_watermark\weights\best_fp16_batch_rtx4090.engine

if not exist "%ENGINE_PATH%" (
    echo [!] FP16 batch engine not found!
    echo Run BUILD_FP16_BATCH.bat first to create it!
    pause
    exit /b 1
)

echo [*] Using FP16 TensorRT batch engine (RTX 4090)
echo [*] Target: 0.7-1.2ms per frame with batch 128 (800-1400 fps)
echo.

"%PYTHON_CMD%" benchmark_fp16_batch.py

echo.
echo ============================================================
echo Benchmark complete!
echo ============================================================
pause
