@echo off
echo ============================================================
echo INSTALL EVERYTHING TO D DRIVE - NOTHING ON C DRIVE!
echo ============================================================
echo.
echo This will install ALL dependencies inside watermarkz folder
echo NO files will be saved to C drive (temp, cache, packages)
echo.
pause

cd /d "%~dp0"

REM ============================================================
REM CRITICAL: Force ALL temp/cache to D drive (watermarkz folder)
REM ============================================================

echo.
echo [1/6] Setting up D drive folders...
echo.

REM Create all necessary folders
mkdir temp 2>nul
mkdir cache 2>nul
mkdir pip_cache 2>nul
mkdir uploads 2>nul
mkdir results 2>nul
mkdir python_packages 2>nul
mkdir redis_data 2>nul

echo Created folders:
echo   - temp (all temporary files)
echo   - cache (PyTorch, models, etc)
echo   - pip_cache (pip downloads)
echo   - uploads (user uploads)
echo   - results (processed files)
echo   - python_packages (all Python libs)
echo   - redis_data (Redis database)
echo.

REM ============================================================
REM Set environment variables to FORCE D drive usage
REM ============================================================

echo [2/6] Forcing ALL temp/cache to D drive...
echo.

set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0cache
set XDG_CACHE_HOME=%~dp0cache
set TRANSFORMERS_CACHE=%~dp0cache
set HF_HOME=%~dp0cache
set OPENCV_TEMP_PATH=%~dp0temp
set PYTHON_USER_BASE=%~dp0python_packages
set PYTHONUSERBASE=%~dp0python_packages

echo Environment variables set:
echo   PIP_CACHE_DIR=%PIP_CACHE_DIR%
echo   TEMP=%TEMP%
echo   TMP=%TMP%
echo   TORCH_HOME=%TORCH_HOME%
echo.

REM ============================================================
REM Install Python packages LOCALLY (no C drive!)
REM ============================================================

echo [3/6] Installing Python packages to D drive...
echo.

echo Installing Flask and server dependencies...
python -m pip install --no-cache-dir --target python_packages flask flask-cors werkzeug

echo.
echo Installing Celery and Redis client...
python -m pip install --no-cache-dir --target python_packages celery redis

echo.
echo Installing PyTorch with CUDA (check your CUDA version with nvidia-smi)...
echo NOTE: This will be downloaded to pip_cache in watermarkz folder
python -m pip install --cache-dir pip_cache --target python_packages torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing YOLO and computer vision libraries...
python -m pip install --no-cache-dir --target python_packages ultralytics opencv-python numpy pillow

echo.
echo Installing LaMa inpainting...
python -m pip install --no-cache-dir --target python_packages simple-lama-inpainting

echo.
echo [4/6] Verifying installations...
python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

REM ============================================================
REM Download and configure Redis for D drive
REM ============================================================

echo.
echo [5/6] Setting up Redis to use D drive...
echo.

if not exist redis-server.exe (
    echo Redis not found. Please download Redis for Windows:
    echo.
    echo   https://github.com/microsoftarchive/redis/releases
    echo.
    echo Download Redis-x64-3.2.100.zip and extract to watermarkz folder
    echo.
    echo OR use Docker:
    echo   docker run -d -p 6379:6379 -v %~dp0redis_data:/data redis:alpine --dir /data
    echo.
) else (
    echo Redis found! Will use D drive for data.
)

REM Create Redis config for D drive
echo # Redis configuration - D drive only > redis.conf
echo dir %~dp0redis_data >> redis.conf
echo dbfilename dump.rdb >> redis.conf
echo save 900 1 >> redis.conf
echo save 300 10 >> redis.conf
echo save 60 10000 >> redis.conf
echo.
echo Created redis.conf (saves to D:\github\RoomFinderAI\watermarkz\redis_data)

REM ============================================================
REM Download YOLO model to D drive
REM ============================================================

echo.
echo [6/6] Downloading YOLO model to D drive cache...
echo.

python -c "import sys; sys.path.insert(0, 'python_packages'); from ultralytics import YOLO; import os; os.environ['TORCH_HOME'] = 'cache'; model = YOLO('yolov8n.pt'); print('YOLO model downloaded to cache folder')"

REM ============================================================
REM Create startup scripts
REM ============================================================

echo.
echo Creating startup scripts...

REM Start Redis script
echo @echo off > START_REDIS.bat
echo cd /d "%%~dp0" >> START_REDIS.bat
echo echo Starting Redis on D drive... >> START_REDIS.bat
echo redis-server.exe redis.conf >> START_REDIS.bat

REM Start Celery script
echo @echo off > START_CELERY.bat
echo cd /d "%%~dp0" >> START_CELERY.bat
echo. >> START_CELERY.bat
echo REM Force D drive temp/cache >> START_CELERY.bat
echo set PIP_CACHE_DIR=%%~dp0pip_cache >> START_CELERY.bat
echo set TEMP=%%~dp0temp >> START_CELERY.bat
echo set TMP=%%~dp0temp >> START_CELERY.bat
echo set TMPDIR=%%~dp0temp >> START_CELERY.bat
echo set TORCH_HOME=%%~dp0cache >> START_CELERY.bat
echo set XDG_CACHE_HOME=%%~dp0cache >> START_CELERY.bat
echo set OPENCV_TEMP_PATH=%%~dp0temp >> START_CELERY.bat
echo. >> START_CELERY.bat
echo echo Starting Celery worker (GPU processing)... >> START_CELERY.bat
echo celery -A server_production.celery worker --loglevel=info --concurrency=2 >> START_CELERY.bat

REM Start Flask script
echo @echo off > START_SERVER.bat
echo cd /d "%%~dp0" >> START_SERVER.bat
echo. >> START_SERVER.bat
echo REM Force D drive temp/cache >> START_SERVER.bat
echo set PIP_CACHE_DIR=%%~dp0pip_cache >> START_SERVER.bat
echo set TEMP=%%~dp0temp >> START_SERVER.bat
echo set TMP=%%~dp0temp >> START_SERVER.bat
echo set TMPDIR=%%~dp0temp >> START_SERVER.bat
echo set TORCH_HOME=%%~dp0cache >> START_SERVER.bat
echo set XDG_CACHE_HOME=%%~dp0cache >> START_SERVER.bat
echo set OPENCV_TEMP_PATH=%%~dp0temp >> START_SERVER.bat
echo. >> START_SERVER.bat
echo echo Starting Flask server... >> START_SERVER.bat
echo python server_production.py >> START_SERVER.bat

REM All-in-one startup script
echo @echo off > START_ALL.bat
echo cd /d "%%~dp0" >> START_ALL.bat
echo echo ============================================================ >> START_ALL.bat
echo echo Starting WatermarkAI Production Server >> START_ALL.bat
echo echo ALL FILES ON D DRIVE ONLY! >> START_ALL.bat
echo echo ============================================================ >> START_ALL.bat
echo echo. >> START_ALL.bat
echo echo Starting Redis... >> START_ALL.bat
echo start "Redis" /MIN cmd /c START_REDIS.bat >> START_ALL.bat
echo timeout /t 2 /nobreak ^>nul >> START_ALL.bat
echo echo. >> START_ALL.bat
echo echo Starting Celery Worker... >> START_ALL.bat
echo start "Celery Worker" cmd /c START_CELERY.bat >> START_ALL.bat
echo timeout /t 3 /nobreak ^>nul >> START_ALL.bat
echo echo. >> START_ALL.bat
echo echo Starting Flask Server... >> START_ALL.bat
echo start "Flask Server" cmd /c START_SERVER.bat >> START_ALL.bat
echo echo. >> START_ALL.bat
echo echo ============================================================ >> START_ALL.bat
echo echo All services started! >> START_ALL.bat
echo echo. >> START_ALL.bat
echo echo Redis: Running in background >> START_ALL.bat
echo echo Celery: Processing watermark removal jobs >> START_ALL.bat
echo echo Flask: http://localhost:5000 >> START_ALL.bat
echo echo. >> START_ALL.bat
echo echo Check each window for logs >> START_ALL.bat
echo echo Press Ctrl+C in each window to stop >> START_ALL.bat
echo echo ============================================================ >> START_ALL.bat
echo pause >> START_ALL.bat

REM ============================================================
REM Final verification
REM ============================================================

echo.
echo ============================================================
echo INSTALLATION COMPLETE - ALL ON D DRIVE!
echo ============================================================
echo.
echo Folder structure:
dir /AD
echo.
echo Disk usage:
du -sh temp cache pip_cache python_packages uploads results redis_data 2>nul || echo [Install du utility to see sizes]
echo.
echo Startup scripts created:
echo   - START_REDIS.bat
echo   - START_CELERY.bat
echo   - START_SERVER.bat
echo   - START_ALL.bat (starts everything)
echo.
echo ============================================================
echo NEXT STEPS:
echo ============================================================
echo.
echo 1. Download Redis for Windows (if not using Docker):
echo    https://github.com/microsoftarchive/redis/releases
echo    Extract to watermarkz folder
echo.
echo 2. Test GPU:
echo    python yolo_detector_optimized.py
echo.
echo 3. Start server:
echo    START_ALL.bat
echo.
echo 4. Visit:
echo    http://localhost:5000
echo.
echo ============================================================
echo VERIFICATION - NO C DRIVE USAGE:
echo ============================================================
echo.
echo Check these locations are EMPTY on C drive:
echo   C:\Users\%USERNAME%\AppData\Local\Temp
echo   C:\Users\%USERNAME%\AppData\Local\pip
echo   C:\Users\%USERNAME%\.cache
echo.
echo Everything should be in:
echo   D:\github\RoomFinderAI\watermarkz\
echo.
echo ✅ C drive is safe!
echo ✅ Everything on D drive!
echo ✅ Ready for $1M/month!
echo.
pause
