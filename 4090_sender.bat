@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo 4090 SENDER - WSL SAM2 Worker
echo ============================================================
echo   - Receives job from website
echo   - Generates masks with SAM2 (PyTorch)
echo   - Uploads masks to B2 CDN
echo   - Chains to 4090_receiver for ProPainter
echo ============================================================
echo.

REM Load Redis URL from file or use default
if exist redis_url.txt (
    for /f "usebackq tokens=*" %%A in ("redis_url.txt") do set REDIS_URL=%%A
    echo [REDIS] Loaded from redis_url.txt
) else (
    set REDIS_URL=redis://:watermarkz_secure_2024@localhost:6379/0
    echo [REDIS] Using default localhost
)

REM B2 credentials for mask upload
set B2_KEY_ID=00539db5c1104b50000000003
set B2_APP_KEY=K005384b8lPoBT11wScxkZ2Gx0fszus
set B2_BUCKET=watermarkz
set B2_CDN_URL=https://markz.humblewoslayer.workers.dev

echo [B2] Upload enabled to %B2_BUCKET%
echo.

REM TGLDP v2: SD-Guided Enhancement for hard regions
set USE_SD_GUIDANCE=1
set SD_INFERENCE_STEPS=3
set SD_HARD_REGION_THRESHOLD=0.15
echo [TGLDP] SD Guidance enabled (v2)
echo.

REM Start WSL Celery worker
echo Starting WSL worker on LOCAL queues: wsl_sam2_local, wsl_yolo_local
echo.

wsl -e bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && export REDIS_URL='%REDIS_URL%' && export B2_KEY_ID='%B2_KEY_ID%' && export B2_APP_KEY='%B2_APP_KEY%' && export B2_BUCKET='%B2_BUCKET%' && export B2_CDN_URL='%B2_CDN_URL%' && export USE_SD_GUIDANCE='%USE_SD_GUIDANCE%' && export SD_INFERENCE_STEPS='%SD_INFERENCE_STEPS%' && export SD_HARD_REGION_THRESHOLD='%SD_HARD_REGION_THRESHOLD%' && celery -A wsl_sam2_worker worker -Q wsl_sam2_local,wsl_yolo_local --loglevel=info --pool=solo"
