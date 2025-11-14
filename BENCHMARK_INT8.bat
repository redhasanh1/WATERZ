@echo off
echo ============================================================
echo INT8 Direct TensorRT Benchmark - EXTREME SPEED TEST
echo ============================================================
echo.

set ENGINE_PATH=runs\detect\new_sora_watermark\weights\best_int8_rtx5070.engine

if not exist "%ENGINE_PATH%" (
    echo [!] INT8 engine not found!
    echo Run BUILD_INT8_ENGINE.bat first!
    pause
    exit /b 1
)

echo [*] Benchmarking INT8 engine with batch 64...
echo [*] Target: 0.7ms per frame (1400+ fps)
echo.

"C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe" yolo_tensorrt_direct.py "%ENGINE_PATH%"

echo.
echo ============================================================
echo Benchmark complete!
echo ============================================================
pause
