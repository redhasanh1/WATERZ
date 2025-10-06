@echo off
cd /d "%~dp0"

echo ================================================================
echo LOCALTUNNEL INSTALLATION
echo Installing portable Node.js + Localtunnel
echo Everything stays in watermarkz folder (NO C drive usage)
echo ================================================================
echo.

REM Check if Node.js is already installed locally
if exist node\node.exe (
    echo [OK] Node.js already installed locally
    goto install_localtunnel
)

echo [1/3] Downloading portable Node.js...
echo.

REM Create node directory
if not exist node mkdir node

REM Download Node.js portable (Windows x64)
echo Downloading Node.js v20.x LTS...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://nodejs.org/dist/v20.11.0/node-v20.11.0-win-x64.zip' -OutFile 'node.zip'}"

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to download Node.js
    echo Please download manually from: https://nodejs.org/dist/v20.11.0/node-v20.11.0-win-x64.zip
    pause
    exit /b 1
)

echo [2/3] Extracting Node.js...
echo.

REM Extract to node folder
powershell -Command "Expand-Archive -Path 'node.zip' -DestinationPath 'node_temp' -Force"
xcopy /E /I /Y node_temp\node-v20.11.0-win-x64\* node\
rmdir /S /Q node_temp
del node.zip

echo [OK] Node.js installed to: %CD%\node\
echo.

:install_localtunnel
echo [3/3] Installing localtunnel...
echo.

REM Set npm to use local folders only
set PATH=%~dp0node;%PATH%
set NPM_CONFIG_PREFIX=%~dp0node_modules
set NPM_CONFIG_CACHE=%~dp0node_cache
set npm_config_prefix=%~dp0node_modules
set npm_config_cache=%~dp0node_cache

REM Create directories
if not exist node_modules mkdir node_modules
if not exist node_cache mkdir node_cache

REM Install localtunnel locally
node\npm install localtunnel

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo SUCCESS!
    echo ================================================================
    echo.
    echo Localtunnel installed to: %CD%\node_modules\
    echo.
    echo Next step: Run START_ALL.bat to start server with tunnel
    echo.
) else (
    echo.
    echo ================================================================
    echo ERROR
    echo ================================================================
    echo.
    echo Failed to install localtunnel
    echo Check your internet connection and try again
    echo.
)

pause
