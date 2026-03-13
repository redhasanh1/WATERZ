#!/bin/bash
# ============================================================
# ProPainter Worker Docker Entrypoint
# ============================================================
# Runs Celery worker in tmux for crash debugging
# ============================================================

set -e

echo "============================================================"
echo "ProPainter Worker - PyTorch (No TensorRT)"
echo "============================================================"

# Check CUDA/GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found or GPU not accessible!"
    echo "        Make sure you're running with --gpus all"
    exit 1
fi

# Detect GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "[GPU] Detected: $GPU_NAME"
echo "[GPU] VRAM: ${GPU_MEMORY}MB"
echo ""

# Check Redis connection
if [ -z "$REDIS_URL" ]; then
    echo "[ERROR] REDIS_URL environment variable not set!"
    exit 1
fi

echo "[REDIS] Connecting to: ${REDIS_URL:0:40}..."

# ============================================================
# Notification helpers
# ============================================================
send_notification() {
    MSG="$1"
    if [ -n "$NOTIFY_WEBHOOK_URL" ]; then
        curl -s -X POST "$NOTIFY_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"$MSG\"}" || true
    fi
}

get_crash_reason() {
    case "$1" in
        0)   echo "Normal shutdown (exit 0)" ;;
        1)   echo "Error (exit 1)" ;;
        137) echo "OOM Killed (exit 137)" ;;
        139) echo "Segfault (exit 139)" ;;
        *)   echo "Unknown (exit $1)" ;;
    esac
}

WORKER_NAME="${NOTIFY_WORKER_NAME:-ProPainter-$(hostname)}"
RESTART_COUNT=0

# ============================================================
# Start Celery in tmux
# ============================================================
echo ""
echo "============================================================"
echo "[START] Launching Celery worker in tmux session"
echo "============================================================"
echo "  Queues: celery, propainter"
echo "  Pool: threads (concurrency 4)"
echo "============================================================"
echo ""

# Kill any existing session
tmux kill-session -t celery 2>/dev/null || true

# Create tmux session with celery
tmux new-session -d -s celery -n "worker" \
    "celery -A server_production2.celery worker -Q celery,propainter --loglevel=info --pool=threads --concurrency=4; echo \$? > /tmp/celery_exit; tmux wait-for -S celery-done"

echo "[OK] Celery started in tmux session 'celery'"
echo ""
echo "  Attach: docker exec -it <container> tmux attach -t celery"
echo "  Detach: Ctrl+B then D"
echo ""

# Send startup notification
send_notification "🟢 Worker **$WORKER_NAME** is READY! GPU: $GPU_NAME (${GPU_MEMORY}MB VRAM)"

# ============================================================
# Monitor and auto-restart on crash
# ============================================================
while true; do
    echo "[MONITOR] Waiting for Celery to exit..."
    tmux wait-for celery-done

    if [ -f /tmp/celery_exit ]; then
        EXIT_CODE=$(cat /tmp/celery_exit)
        rm -f /tmp/celery_exit
    else
        EXIT_CODE=1
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    REASON=$(get_crash_reason $EXIT_CODE)

    echo "[CRASH] Celery exited: $REASON (restart #$RESTART_COUNT)"
    send_notification "🔴 Worker **$WORKER_NAME** crashed! Reason: $REASON | Restarting... (#$RESTART_COUNT)"

    # Clear VRAM before restart
    python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" 2>/dev/null || true

    sleep 3

    # Restart celery in tmux
    tmux kill-session -t celery 2>/dev/null || true
    tmux new-session -d -s celery -n "worker" \
        "celery -A server_production2.celery worker -Q celery,propainter --loglevel=info --pool=threads --concurrency=4; echo \$? > /tmp/celery_exit; tmux wait-for -S celery-done"

    send_notification "🔄 Worker **$WORKER_NAME** restarted (#$RESTART_COUNT)"
done
