@echo off
cd /d "%~dp0"
echo Starting Redis on D drive...
echo.

REM Check if redis-server.exe exists
if exist rediz\redis-server.exe (
    echo Found Redis in rediz folder
    rediz\redis-server.exe redis.conf
) else if exist redis\redis-server.exe (
    echo Found Redis in redis folder
    redis\redis-server.exe redis.conf
) else if exist redis-server.exe (
    echo Found Redis in root folder
    redis-server.exe redis.conf
) else (
    echo ❌ ERROR: redis-server.exe not found!
    echo Please extract Redis to watermarkz\rediz\ folder
    pause
) 
