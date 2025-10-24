@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM  Windows helper for running the Triton fused kernel fallback tests.
REM  This script:
REM    1. Creates (or reuses) a virtual environment at propainter_env_win
REM    2. Upgrades pip
REM    3. Installs the minimum deps required to execute the regression test
REM    4. Runs test\test_triton_fused_kernel.py
REM ==============================================================================

set REPO_ROOT=%~dp0
set VENV_PATH=%REPO_ROOT%propainter_env_win
set ACTIVATE_SCRIPT=%VENV_PATH%\Scripts\activate.bat

if not exist "%VENV_PATH%" (
    echo [INFO] Creating virtual environment in "%VENV_PATH%"
    python -m venv "%VENV_PATH%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Make sure `python` is on PATH.
        exit /b 1
    )
) else (
    echo [INFO] Reusing existing virtual environment at "%VENV_PATH%"
)

call "%ACTIVATE_SCRIPT%"
if errorlevel 1 (
    echo [ERROR] Could not activate the virtual environment.
    exit /b 1
)

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    exit /b 1
)

echo [INFO] Installing dependencies (torch only)...
python -m pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] Failed to install torch. Check CUDA version or use CPU wheel via PyPI.
    exit /b 1
)

echo [INFO] Running regression test...
python test\test_triton_fused_kernel.py
set TEST_ERROR=%errorlevel%

if %TEST_ERROR% neq 0 (
    echo [ERROR] Regression test failed with exit code %TEST_ERROR%.
) else (
    echo [SUCCESS] Triton fused kernel fallback test suite passed.
)

endlocal
exit /b %TEST_ERROR%
