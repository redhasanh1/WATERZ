@echo off
cd /d "%~dp0"

echo ================================================================
echo Installing Waitress Production Server
echo ================================================================
echo.

REM Activate venv if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Installing Waitress...
pip install waitress

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS!
    echo ================================================================
    echo.
    echo Waitress installed successfully
    echo This is a production-ready WSGI server for Flask
    echo.
) else (
    echo.
    echo ================================================================
    echo ERROR
    echo ================================================================
    echo.
    echo Failed to install Waitress
    echo.
)

pause
