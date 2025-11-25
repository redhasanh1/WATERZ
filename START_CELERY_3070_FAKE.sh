#!/bin/bash
# ============================================================================
# FAKE RTX 3070 Processing Worker - For Testing in WSL
# ============================================================================
#
# Simulates ProPainter without GPU - copies frames with green tint
# No GPU required!
#
# ============================================================================

echo "============================================================"
echo "  FAKE RTX 3070 Processing Worker (NO GPU - WSL)"
echo "============================================================"
echo ""

# Change to script directory
cd /mnt/d/watermarkz

echo "Starting FAKE processing worker..."
echo "Queue: processing"
echo ""

python3 -m celery -A celery_3070_fake worker \
    --queues=processing \
    --concurrency=1 \
    --loglevel=info \
    --pool=solo \
    --hostname=fake3070wsl@%h
