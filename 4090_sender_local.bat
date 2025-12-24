@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   SAM2 WORKER - Connected to Production
echo ============================================================
echo   - Production UI: https://markremoverai.com/backgroundremover
echo   - SAM2 tracking via WSL2 Celery worker
echo   - All GPU processing on LOCAL 4090
echo   - Redis connects to production server
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

REM ENABLE B2 UPLOAD - Masks go to B2 for production server access
set SKIP_B2_UPLOAD=0
echo [B2] Mask upload ENABLED - production server can access masks
echo.

REM REVOLUTIONARY STREAMING: DALI GPU-direct video decode
REM Requires: pip install nvidia-dali-cuda120 (in WSL2)
set USE_DALI_STREAMING=1
set SAM2_MAX_HEIGHT=480
echo [DALI] GPU-direct streaming: %USE_DALI_STREAMING%
echo.

REM MASK COMPRESSION: Delta + RLE + zlib (100x smaller than PNGs)
set USE_MASK_COMPRESSION=1
echo [COMPRESS] Mask compression: %USE_MASK_COMPRESSION%
echo.

REM TGLDP v2: SD-Guided Enhancement for hard regions
set USE_SD_GUIDANCE=1
set SD_INFERENCE_STEPS=3
set SD_HARD_REGION_THRESHOLD=0.15
echo [TGLDP] SD Guidance enabled (v2)
echo.

REM PURGE OLD JOBS - Start fresh!
echo [PURGE] Clearing old jobs from queues...
wsl -e bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && export REDIS_URL='%REDIS_URL%' && celery -A wsl_sam2_worker purge -Q wsl_sam2_local,wsl_yolo_local -f"
echo [PURGE] Done - queues cleared!
echo.

REM CLEAR WSL RAM CACHE - Free up garbage collected memory
echo [RAM] Clearing WSL memory cache (not killing processes)...
wsl -e bash -c "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || echo 'Note: Need sudo for cache clear'"
echo [RAM] Cache cleared!
echo.

REM ============================================================
REM NO LOCAL SERVER - Railway serves UI now
REM ============================================================
echo [PRODUCTION] UI served by Railway - no local Flask needed
echo.

REM Start WSL Celery worker
echo [WSL] Starting SAM2 worker on LOCAL queues: wsl_sam2_local, wsl_yolo_local
echo.

wsl -e bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && export REDIS_URL='%REDIS_URL%' && export SKIP_B2_UPLOAD='%SKIP_B2_UPLOAD%' && export USE_DALI_STREAMING='%USE_DALI_STREAMING%' && export SAM2_MAX_HEIGHT='%SAM2_MAX_HEIGHT%' && export USE_MASK_COMPRESSION='%USE_MASK_COMPRESSION%' && export USE_SD_GUIDANCE='%USE_SD_GUIDANCE%' && export SD_INFERENCE_STEPS='%SD_INFERENCE_STEPS%' && export SD_HARD_REGION_THRESHOLD='%SD_HARD_REGION_THRESHOLD%' && celery -A wsl_sam2_worker worker -Q wsl_sam2_local,wsl_yolo_local --loglevel=info --pool=solo"
