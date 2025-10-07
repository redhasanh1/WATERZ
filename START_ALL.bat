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
echo Starting Flask Server (Production Mode with Waitress)...
start "Flask Server" cmd /c START_SERVER_PRODUCTION.bat
echo Waiting for Flask to start...
timeout /t 5 /nobreak >nul
echo.
echo Starting ngrok Tunnel (connects to Flask on port 9000)...
start "ngrok" cmd /c START_NGROK.bat
echo Waiting for tunnel to establish...
timeout /t 8 /nobreak >nul
echo.
echo ============================================================
echo All services started!
echo.
echo ngrok: Check ngrok window for public URL
echo Redis: Running in background
echo Celery: Processing watermark removal jobs
echo Flask: http://localhost:9000
echo.
echo.
echo [IMPORTANT] After ngrok starts:
echo 1. Look for the "Forwarding" line in ngrok window
echo 2. Copy the https://....ngrok-free.app URL
echo 3. Update TUNNEL_URL in Railway with that URL
echo.
echo Check each window for logs
echo Press Ctrl+C in each window to stop
echo ============================================================
pause 
