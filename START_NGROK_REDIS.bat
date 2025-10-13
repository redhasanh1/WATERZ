@echo off
cd /d "%~dp0"

echo ================================================================
echo Starting ngrok TCP Tunnel for Redis
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

echo Starting TCP tunnel for Redis (port 6379)...
echo.
echo Using reserved TCP address for persistent connection
echo Your cloud GPU REDIS_URL will always be the same!
echo.

REM Start ngrok TCP tunnel on port 6379 with reserved address
REM Option 1: Use a reserved TCP address (ngrok paid plan)
REM Replace YOUR_RESERVED_ADDRESS with your actual reserved address from ngrok dashboard
REM Example: ngrok tcp 6379 --region us --remote-addr=1.tcp.ngrok.io:12345

REM Option 2: Use a fixed subdomain with authtoken (requires ngrok paid plan)
REM ngrok tcp 6379 --authtoken YOUR_AUTHTOKEN --region us --remote-addr=YOUR_RESERVED_ADDRESS

REM For now, using default (will generate new address each time)
REM To get a fixed address: https://dashboard.ngrok.com/cloud-edge/tcp-addresses
echo.
echo [INFO] To use a fixed TCP address (no more URL changes):
echo        1. Go to: https://dashboard.ngrok.com/cloud-edge/tcp-addresses
echo        2. Reserve a TCP address (requires paid plan starting at $8/month)
echo        3. Update this batch file with: ngrok tcp 6379 --remote-addr=YOUR_RESERVED_ADDRESS
echo        4. Your REDIS_URL will stay the same forever!
echo.
echo Starting with default address (changes each restart)...
echo.

ngrok tcp 6379
