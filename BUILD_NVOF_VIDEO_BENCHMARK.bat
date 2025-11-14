@echo off
REM Build NVOF Video Simulation Benchmark
REM
REM Simulates processing a 10-second video (300 frames @ 30fps)
REM Shows real-world speedup of NVOF vs RAFT TensorRT

echo ========================================
echo Building NVOF Video Benchmark
echo ========================================
echo.

REM Paths
set "SDK_PATH=%CD%\Optical_Flow_SDK_5.0.7"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

REM Locate Visual Studio
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"
if not exist "%VS_PATH%\vcvars64.bat" (
    echo ERROR: Visual Studio 2022 not found
    pause
    exit /b 1
)

REM Setup Visual Studio environment
call "%VS_PATH%\vcvars64.bat"

echo.
echo Compiling benchmark_nvof_video.cpp...
echo.

REM Compile
cl /nologo /std:c++17 /EHsc ^
   /I"%SDK_PATH%\NvOFInterface" ^
   /I"%CUDA_PATH%\include" ^
   benchmark_nvof_video.cpp ^
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
echo Running NVOF video simulation...
echo.
echo This will simulate processing a 10-second video (300 frames @ 30fps)
echo and compare total processing time vs RAFT TensorRT baseline.
echo.

REM Run the benchmark
benchmark_nvof_video.exe

pause
