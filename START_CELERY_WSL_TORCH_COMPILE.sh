#!/bin/bash
# WSL2/Linux Celery Worker with torch.compile (RFCNet only - 1.5-2x speedup)
# RAFT compile DISABLED (grid_sample incompatible with torch.compile)

cd "$(dirname "$0")"

echo "============================================================"
echo "WSL2 TORCH.COMPILE WORKER"
echo "  - RFCNet: torch.compile enabled (1.5-2x speedup)"
echo "  - RAFT: NO compile (grid_sample incompatible)"
echo "  - CUDA Graphs: DISABLED (prevents crashes)"
echo "============================================================"
echo

# Load Redis URL
if [ -f "redis_url.txt" ]; then
    export REDIS_URL=$(cat redis_url.txt)
    echo "[REDIS] Loaded: $REDIS_URL"
else
    export REDIS_URL="redis://:watermarkz_secure_2024@localhost:6379/0"
    echo "[REDIS] Using default localhost"
fi

# Load Tunnel URL if exists
if [ -f "tunnel_url.txt" ]; then
    export TUNNEL_URL=$(cat tunnel_url.txt)
    export API_BASE_URL="$TUNNEL_URL"
    echo "[TUNNEL] Using: $TUNNEL_URL"
fi

# === TORCH.COMPILE SETTINGS ===
# Enable RFCNet compile (gives speedup)
export USE_TORCH_COMPILE_RAFT=1

# CRITICAL: Disable CUDA graphs (prevents grid_sample crashes)
export TORCH_CUDAGRAPHS=0
export TORCHINDUCTOR_CUDAGRAPHS=0

# Configure compile caches
export TORCHINDUCTOR_CACHE_DIR="$(pwd)/.torch_compile_cache"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR="$(pwd)/.triton_cache"

# Create cache dirs
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# === OPTICAL FLOW SETTINGS ===
export USE_NEUFLOW=1
export FORCE_TRT_RAFT=0

# === RFCNET SETTINGS ===
export RFCNET_TORCHTRT=0
export FORCE_TRT_RFCNET=1

# === FP8 OPTIMIZATIONS ===
export ENABLE_FP8_TRANSFORMER=1
export ENABLE_FP8_ENCODER=1
export ENABLE_FP8_DECODER=1
export ENABLE_FP8_RFCNET=1

# === DCNV4 RFCNET ===
export ENABLE_DCNV4_RFCNET=1

# === SAM2 INTERACTIVE MODE ===
export USE_INTERACTIVE_SAM2=1
export YOLO_REQUIRE_TENSORRT=0

# === SEGMENT SETTINGS ===
export SEGMENT_WORKERS=1
export SAM2_USE_SEGMENTS=1
export SAM2_PARALLEL_SEGMENTS=1
export MAX_SEGMENT_FRAMES=300
export MAX_SEGMENT_PIXELS=178000

# === SEGMENT DETECTION ===
export SAM2_SEGMENT_DETECTION_MODE=full
export SEGMENT_USE_MOTION_DETECTION=1
export SEGMENT_MOTION_THRESHOLD=8
export SEGMENT_MIN_LEN_FULL=3
export SEGMENT_MERGE_GAP_FULL=10

# === SAM2 MASK SETTINGS ===
export SAM2_PROMPT_MODE=point
export SAM2_MASK_DILATION=4
export USE_SAM2_TRACKING=0

echo
echo "============================================================"
echo "CONFIG SUMMARY:"
echo "  torch.compile: ENABLED (RFCNet only)"
echo "  CUDA Graphs: DISABLED"
echo "  NeuFlow: ENABLED"
echo "  FP8: ENABLED"
echo "============================================================"
echo

# Start Celery worker
python3 -m celery -A server_production2.celery worker -Q sam2 --loglevel=info --pool=solo
