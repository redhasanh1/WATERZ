#!/bin/bash
# ============================================================
# SENDER - WSL Celery Worker for Mask Generation
# ============================================================
# Handles both SAM2 (interactive) and YOLO (automatic) modes
# Generates masks → uploads to B2 CDN → chains to receiver
# ============================================================
#
# HOW TO RUN:
#   cd /mnt/d/watermarkz
#   chmod +x sender.sh
#   ./sender.sh
#
# ============================================================

echo ""
echo "============================================================"
echo "SENDER - WSL Mask Generation Worker"
echo "============================================================"
echo "  - SAM2 mode: Interactive selection (2 segment workers)"
echo "  - YOLO mode: Automatic detection (4 segment workers)"
echo "  - Uploads masks to B2 CDN"
echo "  - Chains to Windows receiver for ProPainter"
echo "============================================================"
echo ""

cd /mnt/d/watermarkz

# Activate Python virtual environment
if [ -d "venv_wsl2" ]; then
    source venv_wsl2/bin/activate
    echo "[OK] Activated venv_wsl2"
else
    echo "[ERROR] venv_wsl2 not found!"
    echo "Create it with: python3 -m venv venv_wsl2"
    exit 1
fi

# B2 credentials for mask upload
export B2_KEY_ID="00539db5c1104b50000000003"
export B2_APP_KEY="K005384b8lPoBT11wScxkZ2Gx0fszus"
export B2_BUCKET="watermarkz"
export B2_CDN_URL="https://markz.humblewoslayer.workers.dev"
echo "[B2] Upload enabled to $B2_BUCKET"

# Load Redis URL from file or use default
if [ -f "redis_url.txt" ]; then
    export REDIS_URL=$(cat redis_url.txt)
    echo "[REDIS] Loaded from redis_url.txt"
else
    export REDIS_URL="redis://:watermarkz_secure_2024@localhost:6379/0"
    echo "[REDIS] Using default localhost"
fi

echo ""
echo "Starting Celery worker..."
echo "  Queues: wsl_sam2, wsl_yolo"
echo "  Pool: solo (1 task at a time)"
echo ""

# Start Celery worker listening to both SAM2 and YOLO queues
celery -A wsl_sam2_worker worker -Q wsl_sam2,wsl_yolo --loglevel=info --pool=solo
