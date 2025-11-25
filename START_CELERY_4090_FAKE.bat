@echo off
REM ============================================================================
REM FAKE RTX 4090 Detection Server - For Testing
REM ============================================================================
REM
REM Simulates detection without YOLO - creates fake segments
REM No GPU required!
REM
REM ============================================================================

echo ============================================================
echo   FAKE RTX 4090 Detection Server (NO GPU)
echo ============================================================
echo.

cd /d %~dp0

set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo Using Python: %PYTHON_EXE%
echo.

echo Starting FAKE detection worker...
echo Queue: detection
echo.

%PYTHON_EXE% -m celery -A celery_4090_fake worker ^
    --queues=detection ^
    --concurrency=1 ^
    --loglevel=info ^
    --pool=solo ^
    --hostname=fake4090@%%h

pause
