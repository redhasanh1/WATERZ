@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

REM ============================================================
REM STATIC MASK GENERATOR - GPU Accelerated
REM ============================================================
REM Generates mask PNG from shape parameters using CUDA/PyTorch
REM
REM Usage:
REM   GENERATE_STATIC_MASK.bat <shape_json> <width> <height> <output_path> [dilate]
REM
REM Examples:
REM   GENERATE_STATIC_MASK.bat "{\"type\":\"rectangle\",\"x\":100,\"y\":100,\"width\":200,\"height\":100}" 1920 1080 mask.png
REM   GENERATE_STATIC_MASK.bat "{\"type\":\"ellipse\",\"cx\":500,\"cy\":300,\"rx\":100,\"ry\":50}" 1920 1080 mask.png 5
REM ============================================================

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
)
if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=python
)

REM Set up environment
set PYTHONPATH=%~dp0;%~dp0\python_packages;%PYTHONPATH%
set TORCH_HOME=%~dp0\cache
set CUDA_MODULE_LOADING=LAZY

REM Parse arguments
set SHAPE=%~1
set WIDTH=%2
set HEIGHT=%3
set OUTPUT=%~4
set DILATE=%5

if "%SHAPE%"=="" (
    echo [ERROR] Missing shape JSON parameter
    echo.
    echo Usage: GENERATE_STATIC_MASK.bat ^<shape_json^> ^<width^> ^<height^> ^<output_path^> [dilate]
    echo.
    echo Shape JSON examples:
    echo   Rectangle: {\"type\":\"rectangle\",\"x\":100,\"y\":100,\"width\":200,\"height\":100}
    echo   Ellipse:   {\"type\":\"ellipse\",\"cx\":500,\"cy\":300,\"rx\":100,\"ry\":50}
    echo   Polygon:   {\"type\":\"polygon\",\"points\":[{\"x\":100,\"y\":100},{\"x\":200,\"y\":100},{\"x\":150,\"y\":200}]}
    exit /b 1
)

if "%WIDTH%"=="" (
    echo [ERROR] Missing width parameter
    exit /b 1
)

if "%HEIGHT%"=="" (
    echo [ERROR] Missing height parameter
    exit /b 1
)

if "%OUTPUT%"=="" (
    echo [ERROR] Missing output path parameter
    exit /b 1
)

echo ============================================================
echo STATIC MASK GENERATOR - GPU Accelerated
echo ============================================================
echo [MASK] Shape: %SHAPE%
echo [MASK] Size: %WIDTH% x %HEIGHT%
echo [MASK] Output: %OUTPUT%
if not "%DILATE%"=="" (
    echo [MASK] Dilation: %DILATE% iterations
)
echo.

REM Build command
set CMD="%PYTHON_PATH%" static_mask_generator.py --shape "%SHAPE%" --width %WIDTH% --height %HEIGHT% --output "%OUTPUT%"
if not "%DILATE%"=="" (
    set CMD=%CMD% --dilate %DILATE%
)

REM Execute
%CMD%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [OK] Mask generated successfully!
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo [ERROR] Mask generation failed with code %ERRORLEVEL%
    echo ============================================================
    exit /b %ERRORLEVEL%
)
