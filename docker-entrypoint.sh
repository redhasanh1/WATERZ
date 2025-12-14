#!/bin/bash
# ============================================================
# Docker Entrypoint - Build TensorRT Engines on First Run
# ============================================================
# Builds NeuFlow and RFCNet TRT engines if they don't exist,
# then starts the Celery worker.
# ============================================================

set -e

echo "============================================================"
echo "4090 ProPainter Worker - TensorRT Edition"
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

# Engine paths
NEUFLOW_ENGINE="/app/faster-propainter-main/models/neuflow_things_fp16.engine"
RFCNET_ENGINE="/app/engines/rfcnet/rfcnet_dcnv4_fp16.engine"

# Pre-built WSL engines (built on RTX 4090 WSL2)
NEUFLOW_ENGINE_WSL="/app/faster-propainter-main/models/neuflow_things_fp16.engine.wsl"
RFCNET_ENGINE_WSL="/app/engines/rfcnet/rfcnet_dcnv4_fp16.engine.wsl"

# ONNX paths
NEUFLOW_ONNX="/app/faster-propainter-main/models/neuflow_things.onnx"
RFCNET_ONNX="/app/engines/rfcnet/rfcnet_dcnv4.onnx"

# ============================================================
# Detect WSL and use pre-built engines if available
# ============================================================
IS_WSL=false
if grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
    IS_WSL=true
    echo "[WSL] Detected WSL environment"
fi

# Check if GPU is RTX 4090 (engines were built for this GPU)
IS_4090=false
if echo "$GPU_NAME" | grep -qi "4090"; then
    IS_4090=true
    echo "[GPU] RTX 4090 detected - can use pre-built engines"
fi

# ============================================================
# Build NeuFlow TRT Engine (if missing)
# ============================================================
if [ ! -f "$NEUFLOW_ENGINE" ]; then
    # Check for pre-built WSL engine first
    if [ "$IS_WSL" = true ] && [ "$IS_4090" = true ] && [ -f "$NEUFLOW_ENGINE_WSL" ]; then
        echo "[WSL] Using pre-built NeuFlow engine (skipping 10min build!)"
        cp "$NEUFLOW_ENGINE_WSL" "$NEUFLOW_ENGINE"
        echo "[OK] NeuFlow TRT engine copied from WSL pre-built: $NEUFLOW_ENGINE"
    elif [ -f "$NEUFLOW_ONNX" ]; then
        echo ""
        echo "[BUILD] NeuFlow TRT engine not found, building..."
        echo "        This takes ~10 minutes on first run."
        echo ""

        # Use CUDNN-only tactics (critical! CUBLAS causes issues)
        trtexec \
            --onnx="$NEUFLOW_ONNX" \
            --saveEngine="$NEUFLOW_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --tacticSources=+CUDNN \
            --builderOptimizationLevel=5

        echo "[OK] NeuFlow TRT engine built: $NEUFLOW_ENGINE"
    else
        echo "[WARN] NeuFlow ONNX not found: $NEUFLOW_ONNX"
        echo "       Will use PyTorch RAFT fallback (slower, more VRAM)"
    fi
else
    echo "[OK] NeuFlow TRT engine exists: $NEUFLOW_ENGINE"
fi

# ============================================================
# Build RFCNet DCNv4 TRT Engine (if missing)
# ============================================================
# DCNv4 Plugin path (built during docker build)
DCNV4_PLUGIN="/app/libdcnv4_plugin.so"

if [ ! -f "$RFCNET_ENGINE" ]; then
    # Check for pre-built WSL engine first
    if [ "$IS_WSL" = true ] && [ "$IS_4090" = true ] && [ -f "$RFCNET_ENGINE_WSL" ]; then
        echo "[WSL] Using pre-built RFCNet engine (skipping 5min build!)"
        mkdir -p "$(dirname "$RFCNET_ENGINE")"
        cp "$RFCNET_ENGINE_WSL" "$RFCNET_ENGINE"
        echo "[OK] RFCNet TRT engine copied from WSL pre-built: $RFCNET_ENGINE"
    elif [ -f "$RFCNET_ONNX" ]; then
        echo ""
        echo "[BUILD] RFCNet DCNv4 TRT engine not found, building..."
        echo "        This takes ~5 minutes on first run."
        echo ""

        # Ensure output directory exists
        mkdir -p "$(dirname "$RFCNET_ENGINE")"

        # Check if DCNv4 plugin exists
        if [ -f "$DCNV4_PLUGIN" ]; then
            echo "[PLUGIN] Loading DCNv4 plugin: $DCNV4_PLUGIN"
            # Build RFCNet with DCNv4 plugin loaded
            # Dynamic shape profiles: masked_flows=[B,T,2,H,W], masks=[B,T,1,H,W]
            trtexec \
                --onnx="$RFCNET_ONNX" \
                --saveEngine="$RFCNET_ENGINE" \
                --fp16 \
                --memPoolSize=workspace:4096 \
                --builderOptimizationLevel=5 \
                --plugins="$DCNV4_PLUGIN" \
                --minShapes=masked_flows:1x8x2x256x256,masks:1x8x1x256x256 \
                --optShapes=masked_flows:1x8x2x480x640,masks:1x8x1x480x640 \
                --maxShapes=masked_flows:1x16x2x720x1280,masks:1x16x1x720x1280

            echo "[OK] RFCNet DCNv4 TRT engine built: $RFCNET_ENGINE"
        else
            echo "[WARN] DCNv4 plugin not found: $DCNV4_PLUGIN"
            echo "       Cannot build RFCNet TRT engine without plugin"
            echo "       Will use PyTorch RFCNet fallback (slower)"
        fi
    else
        echo "[WARN] RFCNet ONNX not found: $RFCNET_ONNX"
        echo "       Will use PyTorch RFCNet fallback (slower)"
    fi
else
    echo "[OK] RFCNet DCNv4 TRT engine exists: $RFCNET_ENGINE"
fi

# ============================================================
# Show final configuration
# ============================================================
echo ""
echo "============================================================"
echo "TensorRT Engine Status:"
echo "============================================================"
if [ -f "$NEUFLOW_ENGINE" ]; then
    SIZE=$(du -h "$NEUFLOW_ENGINE" | cut -f1)
    echo "  NeuFlow:  ENABLED (TRT) - $SIZE"
else
    echo "  NeuFlow:  DISABLED (PyTorch RAFT fallback)"
fi

if [ -f "$RFCNET_ENGINE" ]; then
    SIZE=$(du -h "$RFCNET_ENGINE" | cut -f1)
    echo "  RFCNet:   ENABLED (TRT) - $SIZE"
else
    echo "  RFCNet:   DISABLED (PyTorch fallback)"
fi
echo "============================================================"
echo ""

# Check Redis connection
if [ -z "$REDIS_URL" ]; then
    echo "[ERROR] REDIS_URL environment variable not set!"
    echo "        Example: -e REDIS_URL='redis://:password@host:port/0'"
    exit 1
fi

echo "[REDIS] Connecting to: $REDIS_URL"

# Auto-configure concurrency based on VRAM
if [ "$GPU_MEMORY" -ge 20000 ]; then
    CONCURRENCY=${CELERY_CONCURRENCY:-4}
elif [ "$GPU_MEMORY" -ge 10000 ]; then
    CONCURRENCY=${CELERY_CONCURRENCY:-2}
else
    CONCURRENCY=${CELERY_CONCURRENCY:-1}
fi

echo ""
echo "============================================================"
echo "[START] Launching Celery Worker"
echo "============================================================"
echo "  Queues: celery,propainter"
echo "  Pool: threads"
echo "  Concurrency: $CONCURRENCY"
echo "============================================================"
echo ""

# ============================================================
# PRODUCTION: Auto-restart loop with crash notifications
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
        137) echo "OOM Killed (exit 137)" ;;
        139) echo "Segfault (exit 139)" ;;
        134) echo "SIGABRT (exit 134)" ;;
        143) echo "SIGTERM (exit 143)" ;;
        1)   echo "Error (exit 1)" ;;
        *)   echo "Unknown (exit $1)" ;;
    esac
}

WORKER_NAME="${NOTIFY_WORKER_NAME:-$(hostname)}"
RESTART_COUNT=0
MAX_RESTARTS=100  # Prevent infinite loop on persistent errors
RESTART_DELAY=5   # Seconds between restarts

# NOTE: Startup notification is now sent from WITHIN Celery (worker_ready signal)
# This ensures notification fires AFTER TRT engines are built and Celery is truly ready

# Production restart loop - auto-recovers from crashes
while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo ""
    echo "============================================================"
    echo "[PRODUCTION] Starting Celery worker in tmux (restart #$RESTART_COUNT)"
    echo "============================================================"

    # Kill any existing tmux session
    tmux kill-session -t celery 2>/dev/null || true

    # Run Celery inside tmux (allows reconnecting: tmux attach -t celery)
    tmux new-session -d -s celery "celery -A server_production2.celery worker \
        --loglevel=WARNING \
        --pool=${CELERY_POOL:-threads} \
        --concurrency=${CONCURRENCY:-4} \
        -Q celery,propainter; echo '[TMUX] Celery exited with code '\$?; sleep 5"

    echo "[TMUX] Celery running in tmux session 'celery'"
    echo "[TMUX] To view logs: tmux attach -t celery"

    # Wait for tmux session to exit (celery crashed or stopped)
    while tmux has-session -t celery 2>/dev/null; do
        sleep 5
    done

    EXIT_CODE=1  # Assume error if tmux session ended
    RESTART_COUNT=$((RESTART_COUNT + 1))
    REASON=$(get_crash_reason $EXIT_CODE)

    echo ""
    echo "[CRASH] Celery exited with code $EXIT_CODE ($REASON)"
    echo "[CRASH] Restart attempt $RESTART_COUNT/$MAX_RESTARTS"

    # Send crash notification
    send_notification "🔴 Worker **$WORKER_NAME** CRASHED! Reason: $REASON | Restarting in ${RESTART_DELAY}s... (attempt $RESTART_COUNT)"

    # Wait before restart (gives system time to recover from OOM)
    sleep $RESTART_DELAY

    # Clear VRAM before restart
    echo "[CLEANUP] Clearing VRAM before restart..."
    python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect(); print('[CLEANUP] VRAM cleared')" 2>/dev/null || true

    # Send restart notification
    send_notification "🔄 Worker **$WORKER_NAME** restarting... (attempt $RESTART_COUNT)"
done

# If we hit max restarts, something is seriously wrong
send_notification "💀 Worker **$WORKER_NAME** DEAD! Hit max restarts ($MAX_RESTARTS). Manual intervention required."
echo "[FATAL] Hit max restarts ($MAX_RESTARTS). Exiting."
exit 1
