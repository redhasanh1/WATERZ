@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo FP8 CONV2D/CONV3D BENCHMARK
echo Testing Encoder/Decoder + RFC Net FP8 optimizations
echo ============================================================
echo.

REM Activate Visual Studio 2022 C++ environment
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

REM Setup TensorRT environment
call "%~dp0SETUP_TRT_ENV.bat"

REM Set CUDA path
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

REM Use explicit Python path
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

echo.
echo Starting benchmark (this will take 10-15 minutes)...
echo Running 4 tests:
echo   1. FP8 OFF (Baseline)
echo   2. FP8 ON (All components)
echo   3. FP8 Encoder+Decoder only
echo   4. FP8 RFCNet only
echo.

"%PYTHON_PATH%" benchmark_fp8_conv.py

echo.
echo ============================================================
echo BENCHMARK COMPLETE
echo Check the results above for performance comparison
echo ============================================================
pause
