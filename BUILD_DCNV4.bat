@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM Force CUDA 12.6 (PyTorch-supported, VS 2022 compatible)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_PATH_V12_6=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%

REM Activate Visual Studio 2022 Build Tools C++ environment
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set DISTUTILS_USE_SDK=1
    set MSSdk=1
    echo [OK] Visual Studio C++ environment activated
) else (
    echo [WARNING] Visual Studio Build Tools not found at default location
)

echo ============================================================
echo DCNv4 BUILD SCRIPT - RTX 4090 Optimization
echo ============================================================
echo.
echo This script will build DCNv4 (3x faster than DCNv3) for ProPainter.
echo Requires: CUDA Toolkit 12.6 with nvcc compiler (PyTorch cu126 compatible)
echo.

REM Use explicit Python path
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

echo [1/5] Verifying CUDA Toolkit installation...
nvcc --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CUDA compiler 'nvcc' not found in PATH!
    echo.
    echo Please install CUDA Toolkit 12.6 from:
    echo https://developer.nvidia.com/cuda-12-6-0-download-archive
    echo.
    echo After installation, restart this script.
    pause
    exit /b 1
)

echo [OK] CUDA compiler found:
nvcc --version | findstr "release"
echo.

echo [2/5] Verifying PyTorch CUDA support...
"%PYTHON_PATH%" -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')" 2>nul
if errorlevel 1 (
    echo [ERROR] PyTorch with CUDA not found!
    pause
    exit /b 1
)
echo.

echo [3/5] Cloning DCNv4 repository...
if exist DCNv4 (
    echo [INFO] DCNv4 directory already exists, using existing...
) else (
    git clone https://github.com/OpenGVLab/DCNv4.git
    if errorlevel 1 (
        echo [ERROR] Failed to clone DCNv4 repository!
        pause
        exit /b 1
    )
)
echo.

echo [4/5] Building DCNv4 CUDA extension (this takes 3-5 minutes)...
cd DCNv4\DCNv4_op
"%PYTHON_PATH%" setup.py build_ext --inplace
if errorlevel 1 (
    echo [ERROR] DCNv4 build failed!
    cd ..\..
    pause
    exit /b 1
)
cd ..\..
echo.

echo [5/5] Installing DCNv4 package...
"%PYTHON_PATH%" -m pip install -e DCNv4
if errorlevel 1 (
    echo [ERROR] DCNv4 installation failed!
    pause
    exit /b 1
)
echo.

echo ============================================================
echo [SUCCESS] DCNv4 installed successfully!
echo ============================================================
echo.
echo Testing DCNv4 import...
"%PYTHON_PATH%" -c "from DCNv4 import dcnv4_conv; print('[OK] DCNv4 import successful!')"
if errorlevel 1 (
    echo [WARNING] DCNv4 import test failed - check logs above
    pause
    exit /b 1
)
echo.
echo Next steps:
echo 1. Run: python test_dcnv4_integration.py
echo 2. Integrate DCNv4 into BidirectionalPropagation
echo.
pause
