@echo off
cd /d "%~dp0"

echo ======================================================================
echo INT8 vs FP16 TensorRT Benchmark
echo RTX 4090 Performance Comparison
echo ======================================================================
echo.

REM Setup TensorRT environment
call "%~dp0SETUP_TRT_ENV.bat"
if errorlevel 1 (
    echo [ERROR] TensorRT environment setup failed!
    pause
    exit /b 1
)

echo.
echo [*] Running benchmark...
echo.

"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" benchmark_int8_vs_fp16.py

if errorlevel 1 (
    echo.
    echo [!] Benchmark failed!
    pause
    exit /b 1
)

echo.
pause
