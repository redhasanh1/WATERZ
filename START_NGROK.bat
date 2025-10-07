@echo off
cd /d "%~dp0"

echo ================================================================
echo Starting ngrok Tunnel
echo ================================================================
echo.

REM Check if ngrok exists
if not exist ngrok.exe (
    echo [ERROR] ngrok.exe not found!
    echo.
    echo Download from: https://ngrok.com/download
    echo Extract ngrok.exe to: %CD%
    echo.
    pause
    exit /b 1
)

echo Starting tunnel on port 9000...
echo.
echo IMPORTANT: Copy the ngrok URL and update Railway TUNNEL_URL
echo.

REM Start ngrok
ngrok http 9000
