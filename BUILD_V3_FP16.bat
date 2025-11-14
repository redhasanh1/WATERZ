@echo off
echo ===============================================================================
echo Building Transformer v3 (FP16-Constrained)
echo ===============================================================================
echo.

REM Setup TensorRT environment
call SETUP_TRT_ENV.bat
if errorlevel 1 (
    echo Failed to setup TensorRT environment
    pause
    exit /b 1
)

REM Install TensorRT Python wheels if not already installed
echo.
echo Checking TensorRT Python installation...
%LOCALAPPDATA%\Programs\Python\Python312\python.exe -c "import tensorrt" >nul 2>&1
if errorlevel 1 (
    echo TensorRT not installed, installing wheels...
    %LOCALAPPDATA%\Programs\Python\Python312\python.exe -m pip install --quiet "%~dp0TensorRT-10.13.3.9\python\tensorrt-10.13.3.9-cp312-none-win_amd64.whl" "%~dp0TensorRT-10.13.3.9\python\tensorrt_dispatch-10.13.3.9-cp312-none-win_amd64.whl"
    if errorlevel 1 (
        echo Failed to install TensorRT Python wheels
        pause
        exit /b 1
    )
    echo TensorRT installed successfully!
) else (
    echo TensorRT already installed.
)

echo.
echo Starting build...
echo.

REM Run builder with Python 3.12 (same version that has TensorRT installed)
%LOCALAPPDATA%\Programs\Python\Python312\python.exe build_trt_full_transformer_v3_fp16_constrained.py
pause
