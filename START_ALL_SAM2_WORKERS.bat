@echo off
REM ===================================================================
REM SAM2 All Workers Launcher - 4 TRT Clickers + 1 WSL Sender
REM ===================================================================
REM Starts:
REM   - 4x TensorRT SAM2 clicker workers (real-time mask preview)
REM   - 1x WSL2 SAM2 Celery sender (full-video mask generation)
REM ===================================================================

chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ===================================================================
echo SAM2 ALL WORKERS LAUNCHER
echo ===================================================================
echo   4x TensorRT Clickers (Flask+Redis) - ports 5555-5558
echo   1x WSL2 Sender (Celery) - wsl_sam2,wsl_yolo queues
echo ===================================================================
echo.

REM Use Python 3.12 (TensorRT requires Python 3.12, not 3.13!)
set PYTHON=C:\Users\has\AppData\Local\Programs\Python\Python312\python.exe
if not exist "%PYTHON%" set PYTHON=python

REM Set environment
set PYTHONPATH=%CD%

REM Load Redis URL from file or use default
if exist redis_url.txt (
    set /p REDIS_URL=<redis_url.txt
    echo [REDIS] Loaded from redis_url.txt
) else (
    set REDIS_URL=redis://:watermarkz_secure_2024@localhost:6379/0
    echo [REDIS] Using default localhost
)

REM Check if .env file exists
if not exist .env (
    echo [WARN] .env file not found - some features may not work
)

REM Activate Python environment (if using venv)
if exist venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check dependencies
echo [INFO] Checking dependencies...
%PYTHON% -c "import flask" 2>nul
if errorlevel 1 (
    echo [WARN] flask not installed, installing...
    %PYTHON% -m pip install flask
)

%PYTHON% -c "import dotenv" 2>nul
if errorlevel 1 (
    echo [WARN] python-dotenv not installed, installing...
    %PYTHON% -m pip install python-dotenv
)

echo.
echo ===================================================================
echo STARTING 4 TENSORRT CLICKER WORKERS
echo ===================================================================
echo [INFO] Dashboards: http://localhost:5555, 5556, 5557, 5558
echo [INFO] All workers share 1 GPU (time-slice, not parallel)
echo ===================================================================
echo.

REM Start 4 TensorRT clicker workers in separate windows
start "SAM2-TRT-0" cmd /k "%PYTHON% start_object_server.py --worker-id 0 --port 5555"
timeout /t 2 >nul
start "SAM2-TRT-1" cmd /k "%PYTHON% start_object_server.py --worker-id 1 --port 5556"
timeout /t 2 >nul
start "SAM2-TRT-2" cmd /k "%PYTHON% start_object_server.py --worker-id 2 --port 5557"
timeout /t 2 >nul
start "SAM2-TRT-3" cmd /k "%PYTHON% start_object_server.py --worker-id 3 --port 5558"

echo.
echo ===================================================================
echo STARTING WSL2 SAM2 SENDER WORKER
echo ===================================================================
echo [INFO] Queues: wsl_sam2, wsl_yolo
echo [INFO] Generates full-video SAM2 masks and uploads to B2
echo ===================================================================
echo.

REM Start WSL2 SAM2 sender worker
timeout /t 2 >nul
start "SAM2-SENDER-WSL" cmd /k "call 4090_sender.bat"

echo.
echo ===================================================================
echo ALL 5 SAM2 WORKERS STARTED!
echo ===================================================================
echo.
echo   TRT Clicker Dashboards:
echo     - Worker 0: http://localhost:5555
echo     - Worker 1: http://localhost:5556
echo     - Worker 2: http://localhost:5557
echo     - Worker 3: http://localhost:5558
echo.
echo   WSL2 Sender: Check "SAM2-SENDER-WSL" window for logs
echo.
echo   Press Ctrl+C in each window to stop individual workers
echo.
echo ===================================================================
pause
