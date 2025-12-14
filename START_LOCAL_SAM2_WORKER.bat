@echo off
REM ===================================================================
REM SAM2 Selection Worker Launcher - 4 WORKERS
REM Starts 4 workers with unique IDs and ports for load distribution
REM ===================================================================

echo.
echo ===================================================================
echo SAM2 SELECTION WORKERS (4x)
echo ===================================================================
echo.

REM Use Python 3.12 (TensorRT requires Python 3.12, not 3.13!)
set PYTHON=C:\Users\has\AppData\Local\Programs\Python\Python312\python.exe

REM Set environment
set PYTHONPATH=%CD%

REM Check if .env file exists
if not exist .env (
    echo [ERROR] .env file not found!
    echo Please create .env with DATABASE_URL
    pause
    exit /b 1
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
echo [INFO] Starting 4 SAM2 Selection Workers...
echo [INFO] Dashboards: http://localhost:5555, 5556, 5557, 5558
echo [INFO] All workers share 1 GPU (time-slice, not parallel)
echo [INFO] Press Ctrl+C in each window to stop
echo.

REM Set Railway Redis URL
set REDIS_URL=redis://default:bwQmxUCQEXUlYTWACmPbbkpnHPVpoiIa@tramway.proxy.rlwy.net:48930

REM Start 4 workers in separate windows with unique IDs and ports
start "SAM2-Worker-0" cmd /k "%PYTHON% start_object_server.py --worker-id 0 --port 5555"
timeout /t 2 >nul
start "SAM2-Worker-1" cmd /k "%PYTHON% start_object_server.py --worker-id 1 --port 5556"
timeout /t 2 >nul
start "SAM2-Worker-2" cmd /k "%PYTHON% start_object_server.py --worker-id 2 --port 5557"
timeout /t 2 >nul
start "SAM2-Worker-3" cmd /k "%PYTHON% start_object_server.py --worker-id 3 --port 5558"

echo.
echo [OK] All 4 workers started!
echo [INFO] Check dashboards at ports 5555-5558
echo.
pause
