@echo off
cd /d "%~dp0"

REM Set Node.js paths
set PATH=%~dp0node;%PATH%
set NODE_PATH=%~dp0node_modules

echo ================================================================
echo Starting Localtunnel
echo ================================================================
echo.

REM Check if localtunnel is installed
if not exist node_modules\localtunnel (
    echo [ERROR] Localtunnel not installed!
    echo Run INSTALL_LOCALTUNNEL.bat first
    pause
    exit /b 1
)

echo Starting tunnel on port 9000...
echo.

REM Start localtunnel and capture output
REM Use local node_modules bin
set TUNNEL_CMD=%~dp0node_modules\.bin\lt.cmd
if not exist "%TUNNEL_CMD%" (
    set TUNNEL_CMD=%~dp0node\node.exe %~dp0node_modules\localtunnel\bin\lt.js
)

REM Clear old tunnel output
if exist tunnel_output.txt del tunnel_output.txt

REM Run tunnel and capture URL
echo Connecting to localtunnel...
echo.
echo [INFO] Make sure Flask server is running on port 9000 first!
echo [INFO] If you see firewall errors, allow Node.js through Windows Firewall
echo.
start /b %TUNNEL_CMD% --port 9000 > tunnel_output.txt 2>&1

REM Wait for tunnel to establish
timeout /t 8 /nobreak >nul

REM Extract URL from output
echo.
echo Extracting tunnel URL...

REM Read the output file and find the URL
for /f "tokens=*" %%a in ('findstr /C:"your url is:" tunnel_output.txt') do (
    set TUNNEL_LINE=%%a
)

REM Parse the URL from the line
if defined TUNNEL_LINE (
    REM Extract URL (format: "your url is: https://...")
    for /f "tokens=4" %%b in ("%TUNNEL_LINE%") do (
        set TUNNEL_URL=%%b
    )
)

REM Write URL to web folder for frontend
if defined TUNNEL_URL (
    echo %TUNNEL_URL% > web\tunnel_url.txt
    echo.
    echo ================================================================
    echo TUNNEL ACTIVE!
    echo ================================================================
    echo.
    echo Your public URL: %TUNNEL_URL%
    echo Saved to: web\tunnel_url.txt
    echo.
    echo Frontend will auto-detect this URL
    echo Keep this window open!
    echo.
) else (
    echo [ERROR] Could not detect tunnel URL
    echo Check tunnel_output.txt for errors
    echo.
    type tunnel_output.txt
    echo.
    pause
    exit /b 1
)

REM Keep tunnel running
echo Tunnel is running... Press Ctrl+C to stop
echo.
REM Tail the output file to show connection logs
powershell -Command "Get-Content tunnel_output.txt -Wait"
