#!/bin/bash
# WSL2 Parallel Processing Test
# Uses 4 workers to process segments in parallel

cd "$(dirname "$0")"

# Enable 4-worker parallel processing
export MAX_PARALLEL_STREAMS=4
export PARALLEL_MODE=multiprocessing

# Disable torch.compile (avoid JIT compilation issues)
export USE_TORCH_COMPILE_RAFT=0
export RFCNET_TORCHTRT=0

# Enable verbose to see worker progress
export WORKER_VERBOSE=1

echo "========================================="
echo "WSL2 4-Worker Parallel Processing"
echo "========================================="
echo ""
echo "[CONFIG] Workers: $MAX_PARALLEL_STREAMS"
echo "[CONFIG] Mode: $PARALLEL_MODE"
echo ""
echo "Expected behavior:"
echo "  - 4 workers load models in parallel (~5-10s each)"
echo "  - All 4 workers process segments simultaneously"
echo "  - Total time should be ~4x faster than sequential"
echo ""

python3 RUN_SAM2_LOCAL.py
