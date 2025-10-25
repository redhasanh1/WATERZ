@echo off
echo ============================================================
echo Building INT8 TensorRT Engine for EXTREME SPEED
echo Target: 1400+ fps (0.7ms per frame)
echo ============================================================
echo.

REM Install pycuda if needed
echo [*] Checking dependencies...
"C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe" -c "import pycuda" 2>nul
if errorlevel 1 (
    echo [*] Installing pycuda...
    "C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe" -m pip install pycuda
)

echo.
echo [*] Building INT8 engine (this will take 3-5 minutes)...
echo.

"C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe" build_yolo_int8_engine.py

if errorlevel 1 (
    echo.
    echo [!] Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! INT8 engine ready!
echo ============================================================
echo.
echo Next step: Test with direct TensorRT inference
echo Run: BENCHMARK_INT8.bat
echo.
pause
