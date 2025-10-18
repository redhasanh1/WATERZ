#!/bin/bash
#
# Cloud Worker 1 - GPU 0
# Distributed segment processing with threads pool
#
# CRITICAL: Use --pool=threads instead of --pool=solo
# Solo pool blocks task pickup and prevents parallel processing!
#

echo "=================================================="
echo "Starting Cloud Worker 1 (GPU 0)"
echo "=================================================="
echo "Pool Mode: threads (non-blocking)"
echo "Concurrency: 1 (single GPU)"
echo "Tasks: Prepare, Process Segments, Finalize"
echo "=================================================="

# Environment configuration
export SEGMENT_WORKERS=1
export CUDA_VISIBLE_DEVICES=0
export MIN_SEGMENTS=2
export MIN_CHUNK_FRAMES=60

# Redis connection (update with your ngrok tunnel)
export REDIS_URL="${REDIS_URL:-redis://:watermarkz_secure_2024@8.tcp.ngrok.io:17609/0}"

# Celery configuration
export CELERY_BROKER_URL="$REDIS_URL"
export CELERY_RESULT_BACKEND="$REDIS_URL"

# Display configuration
echo ""
echo "Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  MIN_SEGMENTS: $MIN_SEGMENTS"
echo "  MIN_CHUNK_FRAMES: $MIN_CHUNK_FRAMES"
echo "  REDIS_URL: ${REDIS_URL:0:30}..."
echo ""
echo "Starting worker in 3 seconds..."
sleep 3

# Start worker with threads pool
celery -A server_production.celery worker \
  --loglevel=info \
  --pool=threads \
  --concurrency=1 \
  -n worker1@%h \
  --without-gossip \
  --without-mingle \
  --without-heartbeat

# Alternative: with heartbeat for cluster coordination
# celery -A server_production.celery worker \
#   --loglevel=info \
#   --pool=threads \
#   --concurrency=1 \
#   -n worker1@%h
