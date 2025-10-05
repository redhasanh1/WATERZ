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
echo. 
echo ============================================================ 
echo All services started! 
echo. 
echo Redis: Running in background 
echo Celery: Processing watermark removal jobs 
echo Flask: http://localhost:5000 
echo. 
echo Check each window for logs 
echo Press Ctrl+C in each window to stop 
echo ============================================================ 
pause 
