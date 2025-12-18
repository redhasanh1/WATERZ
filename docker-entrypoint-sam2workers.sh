#!/bin/bash
# ============================================================
# SAM2 Workers Docker Entrypoint
# ============================================================
# Runs SAM2 workers in a single container:
#   - N x TensorRT clicker workers (Flask+Redis) on ports 5555+
#   - 1 x Celery sender worker for full-video mask generation
# Features:
#   - tmux session management
#   - Webhook notifications
#   - Unlimited auto-restart
# ============================================================

set -e

echo "============================================================"
echo "SAM2 All Workers - TensorRT Edition"
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

# ============================================================
# SAM2 TensorRT Engine Paths
# ============================================================
ENCODER_ENGINE="/app/sam2_trt_inference/engines/sam2_encoder_fp16.engine"
DECODER_ENGINE="/app/sam2_trt_inference/engines/sam2_decoder_fp16_dynamic.engine"

ENCODER_ONNX="/app/sam2_trt_inference/sam2_pytorch2onnx/output/sam2.1_hiera_tiny_encoder.onnx"
DECODER_ONNX="/app/sam2_trt_inference/sam2_pytorch2onnx/output/sam2.1_hiera_tiny_decoder.onnx"

# Create engines directory
mkdir -p /app/sam2_trt_inference/engines

# ============================================================
# Build SAM2 Encoder TRT Engine (if missing)
# ============================================================
if [ ! -f "$ENCODER_ENGINE" ]; then
    if [ -f "$ENCODER_ONNX" ]; then
        echo ""
        echo "[BUILD] SAM2 Encoder TRT engine not found, building..."
        echo "        This takes ~5 minutes on first run."
        echo ""

        trtexec \
            --onnx="$ENCODER_ONNX" \
            --saveEngine="$ENCODER_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --builderOptimizationLevel=5

        echo "[OK] SAM2 Encoder TRT engine built: $ENCODER_ENGINE"
    else
        echo "[WARN] SAM2 Encoder ONNX not found: $ENCODER_ONNX"
        echo "       Clicker workers will use PyTorch fallback"
    fi
else
    echo "[OK] SAM2 Encoder TRT engine exists: $ENCODER_ENGINE"
fi

# ============================================================
# Build SAM2 Decoder TRT Engine (if missing)
# ============================================================
if [ ! -f "$DECODER_ENGINE" ]; then
    if [ -f "$DECODER_ONNX" ]; then
        echo ""
        echo "[BUILD] SAM2 Decoder TRT engine not found, building..."
        echo "        This takes ~5 minutes on first run."
        echo ""

        trtexec \
            --onnx="$DECODER_ONNX" \
            --saveEngine="$DECODER_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --builderOptimizationLevel=5 \
            --minShapes=point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1 \
            --optShapes=point_coords:4x2x2,point_labels:4x2,mask_input:4x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1 \
            --maxShapes=point_coords:16x2x2,point_labels:16x2,mask_input:16x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1

        echo "[OK] SAM2 Decoder TRT engine built: $DECODER_ENGINE"
    else
        echo "[WARN] SAM2 Decoder ONNX not found: $DECODER_ONNX"
        echo "       Clicker workers will use PyTorch fallback"
    fi
else
    echo "[OK] SAM2 Decoder TRT engine exists: $DECODER_ENGINE"
fi

# ============================================================
# Show final configuration
# ============================================================
echo ""
echo "============================================================"
echo "TensorRT Engine Status:"
echo "============================================================"
if [ -f "$ENCODER_ENGINE" ]; then
    SIZE=$(du -h "$ENCODER_ENGINE" | cut -f1)
    echo "  SAM2 Encoder:  ENABLED (TRT) - $SIZE"
else
    echo "  SAM2 Encoder:  PyTorch fallback"
fi

if [ -f "$DECODER_ENGINE" ]; then
    SIZE=$(du -h "$DECODER_ENGINE" | cut -f1)
    echo "  SAM2 Decoder:  ENABLED (TRT) - $SIZE"
else
    echo "  SAM2 Decoder:  PyTorch fallback"
fi
echo "============================================================"
echo ""

# Check Redis connection
if [ -z "$REDIS_URL" ]; then
    echo "[ERROR] REDIS_URL environment variable not set!"
    echo "        Example: -e REDIS_URL='redis://:password@host:port/0'"
    exit 1
fi

echo "[REDIS] Connecting to: ${REDIS_URL:0:40}..."

# Check B2 credentials
if [ -z "$B2_KEY_ID" ] || [ -z "$B2_APP_KEY" ]; then
    echo "[WARN] B2_KEY_ID or B2_APP_KEY not set - mask uploads will fail!"
    echo "       Pass via: -e B2_KEY_ID=... -e B2_APP_KEY=..."
fi

# ============================================================
# Set environment for Python workers
# ============================================================
export PYTHONPATH=/app:/app/sam2_trt_inference:/app/segment-anything-2
export SAM2_ENCODER_ENGINE="$ENCODER_ENGINE"
export SAM2_DECODER_ENGINE="$DECODER_ENGINE"

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
        2)   echo "Bash misuse (exit 2)" ;;
        126) echo "Permission denied (exit 126)" ;;
        127) echo "Command not found (exit 127)" ;;
        130) echo "SIGINT/Ctrl+C (exit 130)" ;;
        134) echo "SIGABRT (exit 134)" ;;
        137) echo "OOM Killed (exit 137)" ;;
        139) echo "Segfault (exit 139)" ;;
        141) echo "Broken pipe (exit 141)" ;;
        143) echo "SIGTERM (exit 143)" ;;
        *)   echo "Unknown (exit $1)" ;;
    esac
}

WORKER_NAME="${NOTIFY_WORKER_NAME:-SAM2-$(hostname)}"
RESTART_COUNT=0
RESTART_DELAY=5

# ============================================================
# Start ALL workers in ONE tmux session with multiple windows
# ============================================================
NUM_WORKERS=${NUM_WORKERS:-4}

echo ""
echo "============================================================"
echo "[START] Launching all workers in single tmux session"
echo "============================================================"
echo "  Clickers: $NUM_WORKERS (ports 5555-$((5555 + NUM_WORKERS - 1)))"
echo "  Celery: wsl_sam2, wsl_yolo queues"
echo "============================================================"
echo ""

# Kill any existing session
tmux kill-session -t workers 2>/dev/null || true

# Create main tmux session with first clicker (window 0)
if [ "$NUM_WORKERS" -gt 0 ]; then
    tmux new-session -d -s workers -n "clicker_0" \
        "python /app/start_object_server.py --worker-id 0 --port 5555; echo \$? > /tmp/clicker_0_exit; tmux wait-for -S clicker-0-done"
    echo "[WINDOW 0] clicker_0 - port 5555"
    sleep 2

    # Add windows for remaining clickers
    for i in $(seq 1 $((NUM_WORKERS - 1))); do
        PORT=$((5555 + i))
        tmux new-window -t workers -n "clicker_$i" \
            "python /app/start_object_server.py --worker-id $i --port $PORT; echo \$? > /tmp/clicker_\${i}_exit; tmux wait-for -S clicker-\${i}-done"
        echo "[WINDOW $i] clicker_$i - port $PORT"
        sleep 2
    done

    # Add celery sender as last window (with heartbeat disabled)
    CELERY_WINDOW=$NUM_WORKERS
    tmux new-window -t workers -n "celery" \
        "celery -A wsl_sam2_worker worker -Q wsl_sam2,wsl_yolo --loglevel=info --pool=solo --without-heartbeat --without-gossip --without-mingle; echo \$? > /tmp/celery_exit; tmux wait-for -S celery-done"
    echo "[WINDOW $CELERY_WINDOW] celery - wsl_sam2,wsl_yolo queues"
else
    # No clickers - just celery (with heartbeat disabled)
    tmux new-session -d -s workers -n "celery" \
        "celery -A wsl_sam2_worker worker -Q wsl_sam2,wsl_yolo --loglevel=info --pool=solo --without-heartbeat --without-gossip --without-mingle; echo \$? > /tmp/celery_exit; tmux wait-for -S celery-done"
    echo "[WINDOW 0] celery - wsl_sam2,wsl_yolo queues"
fi

echo ""
echo "============================================================"
echo "[OK] All workers started in tmux session 'workers'"
echo "============================================================"
echo "  Attach: docker exec -it <container> tmux attach -t workers"
echo "  Switch windows: Ctrl+B then 0-$NUM_WORKERS"
echo "  Detach: Ctrl+B then D"
echo "============================================================"
echo ""

# Send startup notification
send_notification "🟢 Worker **$WORKER_NAME** is READY! GPU: $GPU_NAME (${GPU_MEMORY}MB VRAM) | Clickers: $NUM_WORKERS"

# ============================================================
# Monitor workers with zero-CPU blocking (tmux wait-for)
# ============================================================
# Build wait channels for all workers
WAIT_CHANNELS="celery-done"
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    WAIT_CHANNELS="$WAIT_CHANNELS clicker-${i}-done"
done

echo "[MONITOR] Waiting for any worker to exit (zero-CPU blocking)..."

# Block until ANY worker exits
# tmux wait-for with multiple channels waits for ALL by default
# We need to wait for any ONE, so we use background + wait -n
for channel in $WAIT_CHANNELS; do
    tmux wait-for $channel &
done
wait -n  # Wait for first one to complete

# Find which worker crashed
CRASHED=""
EXIT_CODE=1
if [ -f /tmp/celery_exit ]; then
    CRASHED="Celery"
    EXIT_CODE=$(cat /tmp/celery_exit)
    rm -f /tmp/celery_exit
fi
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    if [ -f /tmp/clicker_${i}_exit ]; then
        CRASHED="Clicker $i"
        EXIT_CODE=$(cat /tmp/clicker_${i}_exit)
        rm -f /tmp/clicker_${i}_exit
        break
    fi
done

REASON=$(get_crash_reason $EXIT_CODE)
echo "[CRASH] $CRASHED exited: $REASON"
send_notification "🔴 Worker **$WORKER_NAME** - $CRASHED crashed! Reason: $REASON | Container will restart."
exit $EXIT_CODE
