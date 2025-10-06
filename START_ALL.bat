@echo off
cd /d "%~dp0"
echo ============================================================
echo Starting WatermarkAI Production Server
echo ALL FILES ON D DRIVE ONLY!
echo ============================================================
echo.
echo Starting Redis...
start "Redis" /MIN cmd /c START_REDIS.bat
timeout /t 2 /nobreak >nul
echo.
echo Starting Celery Worker...
start "Celery Worker" cmd /c START_CELERY.bat
timeout /t 3 /nobreak >nul
echo.
echo Starting Flask Server...
start "Flask Server" cmd /c START_SERVER.bat
echo Waiting for Flask to start...
timeout /t 5 /nobreak >nul
echo.
echo Starting Localtunnel (connects to Flask on port 9000)...
start "Localtunnel" cmd /c START_LOCALTUNNEL.bat
echo Waiting for tunnel to establish...
timeout /t 8 /nobreak >nul
echo.
echo ============================================================
echo All services started!
echo.
echo Localtunnel: Check Localtunnel window for public URL
echo Redis: Running in background
echo Celery: Processing watermark removal jobs
echo Flask: http://localhost:9000
echo.
if exist web\tunnel_url.txt (
    echo Public URL detected:
    type web\tunnel_url.txt
    echo.
    echo Frontend will auto-connect to this URL
    echo.
)
echo Check each window for logs
echo Press Ctrl+C in each window to stop
echo ============================================================
pause 
