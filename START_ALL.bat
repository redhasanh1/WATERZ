@echo off
setlocal EnableDelayedExpansion
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

echo Detecting public ngrok URL and writing to web\tunnel_url.txt...
REM Query ngrok local API for the HTTPS forwarding URL and persist it for the frontend
powershell -NoProfile -Command "try{ for($i=0;$i -lt 40;$i++){ $r=Invoke-RestMethod -Uri http://127.0.0.1:4040/api/tunnels -ErrorAction SilentlyContinue; if($r){ $u = ($r.tunnels | Where-Object { $_.proto -eq 'https' } | Select-Object -First 1).public_url; if($u){ Set-Content -Path 'web\\tunnel_url.txt' -Value $u -Encoding ASCII; Write-Host ('TUNNEL_URL: ' + $u); break } } Start-Sleep -Milliseconds 500 } } catch { Write-Host 'Could not query ngrok API'; }"
if exist web\tunnel_url.txt (
    for /f "usebackq tokens=*" %%A in ("web\tunnel_url.txt") do set TUNNEL_URL=%%A
    echo TUNNEL_URL detected: !TUNNEL_URL!
) else (
    echo ⚠️  Could not detect ngrok URL automatically. You can paste it into web\tunnel_url.txt manually.
)
echo Starting ngrok TCP Tunnel for Redis (port 6379)...
start "ngrok-redis" cmd /c START_NGROK_REDIS.bat
timeout /t 5 /nobreak >nul
echo.
echo ============================================================
echo All services started!
echo.
echo ngrok: Check ngrok window for public URL
echo Redis ngrok: Check ngrok-redis window for tcp address (REDIS_URL)
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
