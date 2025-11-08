@echo off
cd /d "%~dp0"

echo ======================================================================
echo Building INT8 TensorRT Engine with CALIBRATION (RTX 4090)
echo Target: 1400-2000 fps (1.5-2x faster than FP16!)
echo ======================================================================
echo.

REM Setup TensorRT environment
call "%~dp0SETUP_TRT_ENV.bat"
if errorlevel 1 (
    echo [ERROR] TensorRT environment setup failed!
    exit /b 1
)

echo.
echo [*] Building INT8 calibrated engine (5-10 minutes)...
echo.

"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" build_int8_calibrated.py

if errorlevel 1 (
    echo.
    echo [!] Build failed!
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo SUCCESS! INT8 calibrated engine ready!
echo ======================================================================
echo.
pause
