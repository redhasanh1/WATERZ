@echo off
cd /d "%~dp0"

echo ================================================================
echo WavePaint Simple Test (Debugging Version)
echo ================================================================
echo.
echo This will test each component step-by-step to find the crash.
echo.

REM Force D drive temp/cache
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0cache
set XDG_CACHE_HOME=%~dp0cache
set OPENCV_TEMP_PATH=%~dp0temp
set PYTHONPATH=%~dp0python_packages
set PYTHONUNBUFFERED=1

echo Running diagnostic test...
echo.

python -u test_wavepaint_simple.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================================
    echo Test failed with error code: %ERRORLEVEL%
    echo ================================================================
    echo.
    echo Please share the error message above so I can fix it.
    echo.
) else (
    echo.
    echo ================================================================
    echo Test completed successfully!
    echo ================================================================
    echo.
)

pause
