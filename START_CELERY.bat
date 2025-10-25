@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
REM Minimal environment: only TensorRT-related config
echo Starting Celery worker (TensorRT-optimized)... 

REM Ensure TensorRT DLLs and tools are on PATH
call "%~dp0SETUP_TRT_ENV.bat"

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Force TensorRT-only mode for YOLO (no .pt fallback)
set YOLO_REQUIRE_TENSORRT=1

REM Force TensorRT-only RAFT (no PyTorch fallback)
set FORCE_TRT_RAFT=1

REM Use module form; threads pool works best on Windows
REM concurrency=2 is a safe default for RTX 5070
"%PYTHON_PATH%" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=2 
