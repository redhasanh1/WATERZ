@echo off
REM ===================================================================
REM SAM2 Selection Worker Launcher
REM Starts the local worker that polls Railway for SAM2 selection jobs
REM ===================================================================

echo.
echo ===================================================================
echo SAM2 SELECTION WORKER
echo ===================================================================
echo.

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

REM Check if DATABASE_URL is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); exit(0 if os.getenv('DATABASE_URL') else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] DATABASE_URL not found in .env file!
    echo Please add DATABASE_URL to your .env file
    pause
    exit /b 1
)

REM Check dependencies
echo [INFO] Checking dependencies...
python -c "import psycopg2" 2>nul
if errorlevel 1 (
    echo [WARN] psycopg2 not installed, installing...
    pip install psycopg2-binary
)

python -c "import flask" 2>nul
if errorlevel 1 (
    echo [WARN] flask not installed, installing...
    pip install flask
)

python -c "import dotenv" 2>nul
if errorlevel 1 (
    echo [WARN] python-dotenv not installed, installing...
    pip install python-dotenv
)

echo.
echo [INFO] Starting SAM2 Selection Worker...
echo [INFO] Dashboard will be available at http://localhost:5555
echo [INFO] Press Ctrl+C to stop
echo.

REM Start worker
python start_object_server.py

pause
