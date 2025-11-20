@echo off
REM ============================================================
REM Upload Demo Videos to Railway Volume
REM Run this anytime videos are missing after Railway redeploy
REM ============================================================

echo.
echo ============================================================
echo   UPLOADING DEMO VIDEOS TO RAILWAY
echo ============================================================
echo.

python upload_videos_to_railway.py

echo.
echo ============================================================
echo   DONE! Videos are now live on Railway.
echo ============================================================
echo.
pause
