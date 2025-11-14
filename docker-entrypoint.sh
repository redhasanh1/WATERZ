#!/bin/bash
set -e

echo "=========================================="
echo "  Watermark Removal Worker Starting"
echo "=========================================="
echo ""

# Check CUDA availability
if ! nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found or GPU not accessible!"
    echo "   Make sure you're running with --gpus all"
    exit 1
fi

# Detect GPU model and VRAM
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n 1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo "🎮 Detected GPU: $GPU_NAME"
echo "💾 VRAM: ${GPU_MEMORY}MB"
echo ""

# Auto-configure concurrency based on VRAM
if [ "$GPU_MEMORY" -ge 30000 ]; then
    # RTX 5090, RTX 4090, A6000 (32GB+)
    CONCURRENCY=6
    echo "🚀 High-end GPU detected → concurrency=6"
elif [ "$GPU_MEMORY" -ge 20000 ]; then
    # RTX 4080, RTX 3090 (24GB)
    CONCURRENCY=4
    echo "⚡ Mid-high GPU detected → concurrency=4"
elif [ "$GPU_MEMORY" -ge 10000 ]; then
    # RTX 5070, RTX 3080, RTX 4070 (12-16GB)
    CONCURRENCY=2
    echo "💪 Mid-range GPU detected → concurrency=2"
else
    # Low VRAM GPUs
    CONCURRENCY=1
    echo "⚠️  Low VRAM detected → concurrency=1"
fi

# Allow manual override
if [ -n "$CELERY_CONCURRENCY" ]; then
    CONCURRENCY=$CELERY_CONCURRENCY
    echo "🔧 Manual override: concurrency=$CONCURRENCY"
fi

# Check Redis connection
if [ -z "$REDIS_URL" ]; then
    echo "❌ ERROR: REDIS_URL environment variable not set!"
    echo "   Example: -e REDIS_URL='redis://:password@host:port/0'"
    exit 1
fi

echo "📡 Connecting to Redis: $REDIS_URL"
echo ""

# Set worker name
if [ -z "$WORKER_NAME" ]; then
    WORKER_NAME="worker@$(hostname)"
fi

echo "=========================================="
echo "  Starting Celery Worker"
echo "=========================================="
echo "  Concurrency: $CONCURRENCY"
echo "  Pool: threads"
echo "  Worker Name: $WORKER_NAME"
echo "=========================================="
echo ""

# Start Celery worker
exec python -m celery \
    -A server_production.celery \
    worker \
    --loglevel=info \
    --pool=threads \
    --concurrency=$CONCURRENCY \
    -n "$WORKER_NAME"
