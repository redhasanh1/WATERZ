@echo off
echo ============================================================
echo SAM2 Multi-Worker Launcher
echo ============================================================
echo.

REM Configuration
set NUM_WORKERS=5
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo [ERROR] Python not found at: %PYTHON_PATH%
    echo Please update PYTHON_PATH in this script
    pause
    exit /b 1
)

echo Starting %NUM_WORKERS% SAM2 workers...
echo.
echo Dashboard URLs:
echo   Worker 0: http://localhost:5555
echo   Worker 1: http://localhost:5556
echo   Worker 2: http://localhost:5557
echo   Worker 3: http://localhost:5558
echo   Worker 4: http://localhost:5559
echo.
echo Estimated VRAM usage: 10-20GB (2-4GB per worker)
echo.

REM Launch workers in separate windows
start "SAM2 Worker 0" cmd /c "%PYTHON_PATH% start_object_server.py --worker-id 0"
timeout /t 2 /nobreak >nul

start "SAM2 Worker 1" cmd /c "%PYTHON_PATH% start_object_server.py --worker-id 1"
timeout /t 2 /nobreak >nul

start "SAM2 Worker 2" cmd /c "%PYTHON_PATH% start_object_server.py --worker-id 2"
timeout /t 2 /nobreak >nul

start "SAM2 Worker 3" cmd /c "%PYTHON_PATH% start_object_server.py --worker-id 3"
timeout /t 2 /nobreak >nul

start "SAM2 Worker 4" cmd /c "%PYTHON_PATH% start_object_server.py --worker-id 4"

echo.
echo ============================================================
echo All %NUM_WORKERS% workers launched!
echo ============================================================
echo.
echo Each worker will load its own SAM2 TensorRT model.
echo Wait ~10-20 seconds for all models to load.
echo.
echo To stop all workers, close their terminal windows.
echo.
pause
