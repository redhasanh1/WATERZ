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

REM ✅ LOCAL REDIS ONLY - No cloud Redis tunnel
echo.
echo ✅ Using LOCAL Redis (localhost:6379)

echo.
echo ============================================================
echo All services started!
echo.
echo ✅ ngrok HTTP: !TUNNEL_URL!
echo ✅ Redis: Running locally on localhost:6379
echo ✅ Celery: Processing watermark removal jobs
echo ✅ Flask: http://localhost:9000
echo.
echo Check each window for logs
echo Press Ctrl+C in each window to stop
echo ============================================================
pause 
