@echo off
echo ============================================================
echo DETAILED ERROR REPORT
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking common issues...
echo.

REM Check 1: Flask
echo [1] Checking Flask...
python -c "import sys; sys.path.insert(0, 'python_packages'); import flask; print('   ✅ Flask installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ Flask NOT installed
    echo    Fix: Run INSTALL_ALL_LOCAL.bat
)

REM Check 2: Celery
echo [2] Checking Celery...
python -c "import sys; sys.path.insert(0, 'python_packages'); import celery; print('   ✅ Celery installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ Celery NOT installed
    echo    Fix: Run INSTALL_ALL_LOCAL.bat
)

REM Check 3: Redis client
echo [3] Checking Redis client...
python -c "import sys; sys.path.insert(0, 'python_packages'); import redis; print('   ✅ Redis client installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ Redis client NOT installed
    echo    Fix: Run INSTALL_ALL_LOCAL.bat
)

REM Check 4: YOLO
echo [4] Checking YOLO...
python -c "import sys; sys.path.insert(0, 'python_packages'); from ultralytics import YOLO; print('   ✅ YOLO installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ YOLO NOT installed
    echo    Fix: Run INSTALL_ALL_LOCAL.bat
)

REM Check 5: OpenCV
echo [5] Checking OpenCV...
python -c "import sys; sys.path.insert(0, 'python_packages'); import cv2; print('   ✅ OpenCV installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ OpenCV NOT installed
    echo    Fix: Run INSTALL_ALL_LOCAL.bat
)

REM Check 6: NumPy
echo [6] Checking NumPy...
python -c "import sys; sys.path.insert(0, 'python_packages'); import numpy; print('   ✅ NumPy installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ NumPy NOT installed
    echo    Fix: Run INSTALL_ALL_LOCAL.bat
)

REM Check 7: PyTorch
echo [7] Checking PyTorch...
python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; print('   ✅ PyTorch installed')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo    ❌ PyTorch NOT installed
    echo    Fix: Run INSTALL_CUDA_12.bat
)

REM Check 8: Redis server
echo [8] Checking Redis server...
set REDIS_FOUND=0
if exist redis-server.exe (
    echo    ✅ Redis server found (redis-server.exe)
    set REDIS_FOUND=1
)
if exist redis\redis-server.exe (
    echo    ✅ Redis server found (redis\redis-server.exe)
    set REDIS_FOUND=1
)
if exist rediz\redis-server.exe (
    echo    ✅ Redis server found (rediz\redis-server.exe)
    set REDIS_FOUND=1
)
if %REDIS_FOUND% EQU 0 (
    echo    ❌ Redis server NOT found
    echo    Fix: Download from https://github.com/microsoftarchive/redis/releases
)

REM Check 9: Server files
echo [9] Checking server_production.py...
if exist server_production.py (
    echo    ✅ server_production.py exists
) else (
    echo    ❌ server_production.py missing!
)

echo [10] Checking yolo_detector_optimized.py...
if exist yolo_detector_optimized.py (
    echo    ✅ yolo_detector_optimized.py exists
) else (
    echo    ❌ yolo_detector_optimized.py missing!
)

echo.
echo ============================================================
echo QUICK FIXES
echo ============================================================
echo.
echo If you see Flask/Celery/YOLO/OpenCV errors:
echo    Run: INSTALL_ALL_LOCAL.bat
echo.
echo If you see PyTorch errors:
echo    Run: INSTALL_CUDA_12.bat
echo.
echo If you see Redis server error:
echo    Download: https://github.com/microsoftarchive/redis/releases/download/win-3.2.100/Redis-x64-3.2.100.zip
echo    Extract redis-server.exe to: %~dp0
echo.
pause
