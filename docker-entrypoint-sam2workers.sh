#!/bin/bash
# ============================================================
# SAM2 Workers Docker Entrypoint
# ============================================================
# Runs 5 SAM2 workers in a single container:
#   - 4x TensorRT clicker workers (Flask+Redis) on ports 5555-5558
#   - 1x Celery sender worker for full-video mask generation
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

# ============================================================
# Build SAM2 Encoder TRT Engine (if missing)
# ============================================================
if [ ! -f "$ENCODER_ENGINE" ]; then
    if [ -f "$ENCODER_ONNX" ]; then
        echo ""
        echo "[BUILD] SAM2 Encoder TRT engine not found, building..."
        echo "        This takes ~5 minutes on first run."
        echo ""

        mkdir -p "$(dirname "$ENCODER_ENGINE")"

        trtexec \
            --onnx="$ENCODER_ONNX" \
            --saveEngine="$ENCODER_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --builderOptimizationLevel=5

        echo "[OK] SAM2 Encoder TRT engine built: $ENCODER_ENGINE"
    else
        echo "[ERROR] SAM2 Encoder ONNX not found: $ENCODER_ONNX"
        echo "        Cannot start clicker workers without encoder"
        exit 1
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

        mkdir -p "$(dirname "$DECODER_ENGINE")"

        # All inputs must be specified - fixed + dynamic
        # Fixed: image_embed, high_res_feats_0, high_res_feats_1, has_mask_input
        # Dynamic (batch N): point_coords, point_labels, mask_input
        # NOTE: Using Windows-matching batch sizes (1/64/128) for consistent mask quality
        trtexec \
            --onnx="$DECODER_ONNX" \
            --saveEngine="$DECODER_ENGINE" \
            --fp16 \
            --memPoolSize=workspace:4096 \
            --builderOptimizationLevel=5 \
            --minShapes=point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1 \
            --optShapes=point_coords:64x2x2,point_labels:64x2,mask_input:64x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1 \
            --maxShapes=point_coords:128x2x2,point_labels:128x2,mask_input:128x1x256x256,image_embed:1x256x64x64,high_res_feats_0:1x32x256x256,high_res_feats_1:1x64x128x128,has_mask_input:1

        echo "[OK] SAM2 Decoder TRT engine built: $DECODER_ENGINE"
    else
        echo "[ERROR] SAM2 Decoder ONNX not found: $DECODER_ONNX"
        echo "        Cannot start clicker workers without decoder"
        exit 1
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
    echo "  SAM2 Encoder:  MISSING"
fi

if [ -f "$DECODER_ENGINE" ]; then
    SIZE=$(du -h "$DECODER_ENGINE" | cut -f1)
    echo "  SAM2 Decoder:  ENABLED (TRT) - $SIZE"
else
    echo "  SAM2 Decoder:  MISSING"
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

# ============================================================
# Set environment for Python workers
# ============================================================
export PYTHONPATH=/app:/app/sam2_trt_inference
export SAM2_ENCODER_ENGINE="$ENCODER_ENGINE"
export SAM2_DECODER_ENGINE="$DECODER_ENGINE"

# ============================================================
# Start 4 TensorRT Clicker Workers (background)
# ============================================================
echo ""
echo "============================================================"
echo "[START] Launching 4 TensorRT Clicker Workers"
echo "============================================================"
echo "  Ports: 5555, 5556, 5557, 5558"
echo "  Mode: Flask+Redis pub/sub"
echo "============================================================"
echo ""

# Start clicker workers in background
for i in 0 1 2 3; do
    PORT=$((5555 + i))
    echo "[WORKER $i] Starting on port $PORT..."
    python /app/start_object_server.py --worker-id $i --port $PORT &
    sleep 2  # Stagger GPU loading
done

echo ""
echo "[OK] 4 Clicker workers started in background"
echo ""

# ============================================================
# Start Celery Sender Worker (foreground)
# ============================================================
echo ""
echo "============================================================"
echo "[START] Launching Celery Sender Worker"
echo "============================================================"
echo "  Queues: wsl_sam2, wsl_yolo"
echo "  Pool: solo (CUDA-safe)"
echo "============================================================"
echo ""

# Check B2 credentials
if [ -z "$B2_KEY_ID" ] || [ -z "$B2_APP_KEY" ]; then
    echo "[WARN] B2_KEY_ID or B2_APP_KEY not set - mask uploads will fail!"
    echo "       Pass via: -e B2_KEY_ID=... -e B2_APP_KEY=..."
fi
export B2_BUCKET="${B2_BUCKET:-watermarkz}"
export B2_CDN_URL="${B2_CDN_URL:-https://markz.humblewoslayer.workers.dev}"

# Run Celery sender in foreground (keeps container alive)
exec celery -A wsl_sam2_worker worker \
    -Q wsl_sam2,wsl_yolo \
    --loglevel=info \
    --pool=solo
