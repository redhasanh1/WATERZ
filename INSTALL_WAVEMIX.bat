@echo off
cd /d "%~dp0"

echo ================================================================
echo Installing WaveMix for WavePaint
echo ================================================================
echo.

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache

echo Installing wavemix package...
echo.

python -m pip install wavemix --target python_packages

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS!
    echo ================================================================
    echo.
    echo WaveMix installed successfully
    echo Now you can run TEST_WAVEPAINT_ONLY.bat
    echo.
) else (
    echo.
    echo ================================================================
    echo ERROR
    echo ================================================================
    echo.
    echo Installation failed
    echo.
)

pause
