@echo off
echo ============================================================
echo FP16 TensorRT Batch Benchmark - EXTREME SPEED!
echo No pycuda needed!
echo ============================================================
echo.

set ENGINE_PATH=runs\detect\new_sora_watermark\weights\best_fp16.engine

if not exist "%ENGINE_PATH%" (
    echo [!] FP16 engine not found!
    echo Run BUILD_INT8_ENGINE.bat first to create it!
    pause
    exit /b 1
)

echo [*] Using FP16 TensorRT engine with batch processing
echo [*] Target: 1-2ms per frame (500-1000 fps)
echo.

"C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe" benchmark_fp16_batch.py

echo.
echo ============================================================
echo Benchmark complete!
echo ============================================================
pause
