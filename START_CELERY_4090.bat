@echo off
REM ============================================================================
REM RTX 4090 Detection Server - Celery Worker
REM ============================================================================
REM
REM This runs on the cloud 4090 (Railway) and handles:
REM   - Video detection (YOLO or SAM2)
REM   - Segment job creation for 3070 workers
REM
REM Queue: detection
REM Concurrency: 1 (one detection at a time)
REM ============================================================================

echo ============================================================
echo   RTX 4090 Detection Server
echo ============================================================
echo.

REM Change to script directory
cd /d %~dp0

REM Set Python path (Railway uses system Python)
set PYTHON_EXE=python

REM Check if we're on Windows with custom Python path
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
)

echo Using Python: %PYTHON_EXE%
echo.

REM Start Celery worker
echo Starting detection worker...
echo Queue: detection
echo Concurrency: 1
echo.

%PYTHON_EXE% -m celery -A celery_4090_detection worker ^
    --queues=detection ^
    --concurrency=1 ^
    --loglevel=info ^
    --pool=solo ^
    --hostname=detection@%%h

pause
