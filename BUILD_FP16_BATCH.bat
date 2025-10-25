@echo off
echo ============================================================
echo Building FP16 TensorRT Batch Engine
echo Supports batch 1-128 for EXTREME SPEED
echo ============================================================
echo.

echo [*] Building FP16 batch engine (3-5 minutes)...
echo.

"C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe" build_fp16_batch_engine.py

if errorlevel 1 (
    echo.
    echo [!] Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! FP16 batch engine ready!
echo ============================================================
echo.
echo Next: Run BENCHMARK_FP16_BATCH.bat to test performance
echo Expected: 1-2ms per frame with batch 32/64
echo.
pause
