@echo off
REM Build NVOF Hardware Optical Flow Benchmark
REM
REM This benchmark will definitively answer: Is NVOF faster than RAFT TensorRT?
REM
REM Expected outcomes:
REM   1. NVOF < 3.64ms: Hardware OF is faster → proceed with Python integration
REM   2. NVOF >= 3.64ms: RAFT TensorRT is faster → skip NVOF, focus on RFCNet TensorRT

echo ========================================
echo Building NVOF Benchmark
echo ========================================
echo.

REM Paths
set "SDK_PATH=%CD%\Optical_Flow_SDK_5.0.7"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

REM Locate Visual Studio
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"
if not exist "%VS_PATH%\vcvars64.bat" (
    echo ERROR: Visual Studio 2022 not found at expected path
    echo Please update VS_PATH in this script to match your installation
    pause
    exit /b 1
)

REM Setup Visual Studio environment
call "%VS_PATH%\vcvars64.bat"

echo.
echo Compiling benchmark_nvof.cpp...
echo.

REM Compile using low-level NVOF C API (similar to hardware detection test)
cl /nologo /std:c++17 /EHsc ^
   /I"%SDK_PATH%\NvOFInterface" ^
   /I"%CUDA_PATH%\include" ^
   benchmark_nvof.cpp ^
   /link ^
   /LIBPATH:"%CUDA_PATH%\lib\x64" ^
   cuda.lib

if errorlevel 1 (
    echo.
    echo ERROR: Compilation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build successful!
echo ========================================
echo.
echo Running NVOF benchmark...
echo.
echo This will compare NVOF hardware performance vs RAFT TensorRT (3.64ms baseline)
echo.

REM Run the benchmark
benchmark_nvof.exe

set BENCH_RESULT=%ERRORLEVEL%

echo.
if %BENCH_RESULT%==0 (
    echo ========================================
    echo DECISION: NVOF is FASTER - proceed with Python integration
    echo ========================================
) else (
    echo ========================================
    echo DECISION: RAFT TensorRT is FASTER - skip NVOF integration
    echo ========================================
)

pause
