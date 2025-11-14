@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo DIRECT WATERMARK REMOVAL TEST (NO CELERY)
echo ============================================================
echo.

REM Activate Visual Studio 2022 C++ environment
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    echo [OK] Visual Studio C++ environment activated
)

REM Ensure TensorRT DLLs and tools are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM Use explicit Python path
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Environment flags
set YOLO_REQUIRE_TENSORRT=1
set FORCE_TRT_RAFT=1
set USE_NEUFLOW=0
set RFCNET_TORCHTRT=0
set ENABLE_FLASH_ATTENTION=1

echo.
echo CONFIG:
echo   - TensorRT RAFT: ENABLED
echo   - Flash Attention: ENABLED
echo   - neighbor_length: 10, raft_iter: 10
echo.

if "%~1"=="" (
    echo ERROR: No input video specified!
    echo Usage: TEST_DIRECT.bat input_video.mp4 [output_video.mp4]
    echo.
    pause
    exit /b 1
)

echo Input video: %~1
if "%~2"=="" (
    echo Output video: output_direct_test.mp4
    "%PYTHON_PATH%" test_watermark_direct.py "%~1"
) else (
    echo Output video: %~2
    "%PYTHON_PATH%" test_watermark_direct.py "%~1" "%~2"
)

echo.
pause
