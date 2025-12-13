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

# ONNX paths
NEUFLOW_ONNX="/app/faster-propainter-main/models/neuflow_things.onnx"
RFCNET_ONNX="/app/engines/rfcnet/rfcnet_dcnv4.onnx"

# ============================================================
# Build NeuFlow TRT Engine (if missing)
# ============================================================
if [ ! -f "$NEUFLOW_ENGINE" ]; then
    echo ""
    echo "[BUILD] NeuFlow TRT engine not found, building..."
    echo "        This takes ~10 minutes on first run."
    echo ""

    if [ -f "$NEUFLOW_ONNX" ]; then
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
    echo ""
    echo "[BUILD] RFCNet DCNv4 TRT engine not found, building..."
    echo "        This takes ~5 minutes on first run."
    echo ""

    if [ -f "$RFCNET_ONNX" ]; then
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

# Single worker handles BOTH modes:
# - YOLO mode: celery queue -> forwards to server_production
# - SAM2 mode: propainter queue -> handled directly
exec celery -A server_production2.celery worker \
    -Q celery,propainter \
    --loglevel=info \
    --pool=threads \
    --concurrency=$CONCURRENCY
