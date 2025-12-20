@echo off
echo ============================================================
echo SAM2 DEMO - Local Test
echo ============================================================
echo.
echo Starting SAM2 demo backend + frontend...
echo After startup, open: http://localhost:7262
echo.
echo Upload your video, click on the person, then Track!
echo ============================================================
echo.

cd /d "D:\watermarkz\segment-anything-2\demo"
docker compose up --build
