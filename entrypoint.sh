#!/bin/bash
# ===========================================
# Celery Worker Entrypoint
# GPU detection and environment setup
# ===========================================

set -e

echo "=========================================="
echo "Watermark Removal Celery Worker"
echo "PyTorch Mode (Universal GPU Support)"
echo "=========================================="

# ----- Validate Required Environment Variables -----
if [ -z "$REDIS_URL" ] && [ -z "$CELERY_BROKER_URL" ]; then
    echo "ERROR: REDIS_URL or CELERY_BROKER_URL must be set"
    echo "Example: REDIS_URL=redis://default:password@host:6379/0"
    exit 1
fi

# Use REDIS_URL for both broker and backend if CELERY vars not set
if [ -z "$CELERY_BROKER_URL" ]; then
    export CELERY_BROKER_URL="$REDIS_URL"
fi
if [ -z "$CELERY_RESULT_BACKEND" ]; then
    export CELERY_RESULT_BACKEND="$REDIS_URL"
fi

# ----- GPU Detection -----
echo ""
echo "Detecting GPU..."

if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "Unknown")
    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
    GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
    DRIVER_VERSION=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)

    echo "GPU: $GPU_NAME"
    echo "Memory: $GPU_MEMORY"
    echo "Driver: $DRIVER_VERSION"

    # Check CUDA availability in PyTorch
    CUDA_AVAILABLE=$(python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")

    if [ "$CUDA_AVAILABLE" = "yes" ]; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "Unknown")
        echo "PyTorch CUDA: $CUDA_VERSION"
        echo "GPU Status: Ready"
    else
        echo "WARNING: PyTorch cannot access CUDA"
        echo "Check NVIDIA Container Toolkit installation"
    fi
else
    echo "WARNING: nvidia-smi not found"
    echo "Running in CPU-only mode (very slow)"
fi

# ----- Create Runtime Directories -----
echo ""
echo "Setting up directories..."

mkdir -p /data/temp /data/cache /data/uploads /data/results /data/pip_cache

# Ensure proper permissions
chmod -R 777 /data 2>/dev/null || true

# ----- PyTorch Optimizations -----
echo ""
echo "Configuring PyTorch..."

# Disable TensorRT (PyTorch-only mode)
export YOLO_REQUIRE_TENSORRT=0
export FORCE_TRT_RFCNET=0
export FORCE_TRT_TRANSFORMER=0
export FORCE_TRT_RAFT=0

# Enable torch.compile caching
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=/data/cache/torch_compile

# Performance settings
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "TensorRT: Disabled (PyTorch mode)"
echo "torch.compile: Enabled with caching"

# ----- Test Model Loading (Optional) -----
if [ "${TEST_MODELS:-0}" = "1" ]; then
    echo ""
    echo "Testing model loading..."
    python -c "
import torch
from ultralytics import YOLO

# Test YOLO
print('Loading YOLO model...')
model = YOLO('runs/detect/new_sora_watermark/weights/best.pt')
print('YOLO: OK')

# Test CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(1, 3, 640, 640).to(device)
    print('CUDA Tensor: OK')
else:
    print('CUDA: Not available')
"
    echo "Model test complete"
fi

# ----- Print Configuration -----
echo ""
echo "Configuration:"
echo "  Broker: ${CELERY_BROKER_URL%%@*}@..."
echo "  Pool: ${CELERY_WORKER_POOL:-solo}"
echo "  Concurrency: ${CELERY_CONCURRENCY:-1}"
echo ""

# ----- Start Application -----
echo "Starting Celery worker..."
echo "=========================================="

exec "$@"
