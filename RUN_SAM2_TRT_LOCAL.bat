@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo SAM2 + TensorRT ProPainter LOCAL (Same as working YOLO version)
echo.
echo ============================================================
echo SAME CONFIG AS START_CELERY_TRT_LOCAL.bat BUT WITH SAM2 MASKS
echo ============================================================
echo.

REM Activate Visual Studio 2022 C++ environment (required for torch.compile)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo [OK] Visual Studio C++ environment activated
)

REM Ensure TensorRT DLLs and tools are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM Use explicit Python path
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Use NeuFlow v2 ONNX (10-70x faster than RAFT!)
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=1

REM Flash attention
set ENABLE_FLASH_ATTENTION=1

REM Segment workers
set SEGMENT_WORKERS=4

echo.
echo ============================================================
echo CONFIG (SAME AS WORKING YOLO VERSION):
echo   - Optical Flow: NeuFlow v2 ONNX
echo   - Flash Attention: ENABLED
echo   - Masks: temp_sam2_masks_expanded/
echo ============================================================
echo.

"%PYTHON_PATH%" test_local_sam2_trt.py

pause
