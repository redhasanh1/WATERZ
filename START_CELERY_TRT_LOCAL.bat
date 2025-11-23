@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo Starting LOCAL SAM2 Watermark Removal with torch.compile...
echo.
echo ============================================================
echo LOCAL WATERMARK REMOVAL WITH TORCH.COMPILE
echo Visual Studio C++ environment enabled for 1.5x speedup
echo ============================================================
echo.

REM Activate Visual Studio 2022 C++ environment (required for torch.compile)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo [OK] Visual Studio C++ environment activated
) else (
    echo [WARNING] Visual Studio 2022 not found - torch.compile may fail
)

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Use PyTorch YOLO (TensorRT engine was deleted)
set YOLO_REQUIRE_TENSORRT=0

REM Enable torch.compile for RAFT (1.5-2x speedup after first compile)
set USE_TORCH_COMPILE_RAFT=1

REM Use PyTorch RAFT (NeuFlow engine not available)
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=0

REM Disable RFCNet torch.compile (requires Triton which isn't available on Windows)
set RFCNET_TORCHTRT=0

echo.
echo ============================================================
echo OPTIMIZED CONFIG (RTX 4090):
echo   - YOLO: PyTorch
echo   - RAFT: PyTorch + torch.compile (1.5-2x speedup)
echo   - RFCNet: PyTorch (no torch.compile)
echo ============================================================
echo.

REM Run SAM2 local pipeline with torch.compile enabled
"%PYTHON_PATH%" RUN_SAM2_LOCAL.py

pause
