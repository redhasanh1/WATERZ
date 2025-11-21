#!/bin/bash

# Navigate to script directory
cd "$(dirname "$0")"

# ✅ AUTO-LOAD REDIS_URL from redis_url.txt
REDIS_CONFIG_FILE="$(pwd)/redis_url.txt"
echo "[DEBUG] Looking for redis_url.txt at: $REDIS_CONFIG_FILE"

if [ -f "$REDIS_CONFIG_FILE" ]; then
    echo "[DEBUG] File exists, reading..."
    export REDIS_URL=$(cat "$REDIS_CONFIG_FILE")
    echo "[REDIS] ✅ Loaded from redis_url.txt"
    echo "[REDIS] Value: $REDIS_URL"
else
    echo "[REDIS] ❌ redis_url.txt not found at: $REDIS_CONFIG_FILE"
    echo "[REDIS] Using default localhost fallback"
    export REDIS_URL="redis://:watermarkz_secure_2024@localhost:6379/0"
fi

echo "[DEBUG] Final REDIS_URL = $REDIS_URL"
echo

# ✅ AUTO-LOAD TUNNEL_URL from web/tunnel_url.txt for Railway upload
if [ -f "web/tunnel_url.txt" ]; then
    export TUNNEL_URL=$(cat "web/tunnel_url.txt")
    echo "[UPLOAD] Loaded TUNNEL_URL from web/tunnel_url.txt: $TUNNEL_URL"
    echo "[UPLOAD] ✅ Auto-upload to Railway ENABLED"
else
    echo "[UPLOAD] ⚠️  web/tunnel_url.txt not found - Railway upload disabled"
    echo "[UPLOAD] 💡 Create web/tunnel_url.txt with your Railway URL to enable auto-upload"
fi

# Enable auto-upload by default (set to 0 to disable)
export UPLOAD_RESULT_BACK=${UPLOAD_RESULT_BACK:-1}

# Debug: Show all upload-related environment variables
echo "[DEBUG] TUNNEL_URL = $TUNNEL_URL"
echo "[DEBUG] API_BASE_URL = $API_BASE_URL"
echo "[DEBUG] UPLOAD_RESULT_BACK = $UPLOAD_RESULT_BACK"
echo

echo "Starting Celery worker (TensorRT NeuFlow v2 FP16 - 3-4x faster than ONNX!)..."
echo
echo "============================================================"
echo "TENSORRT MODE: NeuFlow v2 TensorRT FP16 (4 execution contexts, pure GPU)"
echo "============================================================"
echo

# ⚡ WSL2 MODE: Use PyTorch models (TensorRT engines are Windows-only!)
#    TensorRT engines built on Windows don't work on Linux
#    Using cross-platform PyTorch/.pt models instead
export YOLO_REQUIRE_TENSORRT=0

# Use NeuFlow v2 ONNX (10-70x faster than RAFT! - ONNX is cross-platform)
export FORCE_TRT_RAFT=0
export USE_NEUFLOW=1

# ⚡ RFCNET PYTORCH: Use PyTorch RFCNet (cross-platform)
#    TensorRT RFCNet engine is Windows-only
#    PyTorch version still has DCNv4 + FP8 optimizations
export RFCNET_TORCHTRT=0
export FORCE_TRT_RFCNET=0

# ⚡ TRANSFORMER TENSORRT: 5-10x speedup on feature propagation (2.39s → 0.24-0.48s per segment!)
#    Uses pre-built FP16 TensorRT engine for Sparse Temporal Transformer (8 layers, 4 heads)
#    Current: 2.39s per segment (45-65% of pipeline) with PyTorch FP8 + Flash Attention
#    Target:  0.24-0.48s per segment (5-10x faster!) - transformer no longer bottleneck
#    NO FALLBACK: TensorRT-only mode (will fail if engine missing)
#    Build engine with: BUILD_TRANSFORMER_ENGINE.bat
export FORCE_TRT_TRANSFORMER=0

# ❌ SAGEATTENTION: DISABLED (incompatible with ProPainter's sparse transformer!)
#    ProPainter uses grouped-query attention with window-based sparsity
#    SageAttention only supports standard dense attention (BERT/GPT/ViT)
#    Manual attention is FASTER: 0.99-1.02s vs 4.83-5.22s with SageAttention fallback
export ENABLE_SAGE_ATTENTION=0
export SAGE_CUDA_ARCH=89

# ⚡ FLASH ATTENTION: DISABLED (manual attention is faster for sparse transformers!)
#    ProPainter's optimized manual attention: 0.99-1.02s feature propagation
#    Flash Attention/SDPA fallback: 4.83-5.22s (5x slower!)
export ENABLE_FLASH_ATTENTION=0

# ⚡ FP8 TRANSFORMER: 1.3-1.5x speedup on Linear layers (RTX 4090 Ada Lovelace 4th Gen Tensor Cores!)
#    Combined with DCNv4 + Flash Attention = 5-10x faster transformers!
export ENABLE_FP8_TRANSFORMER=1

# ❌ TOKEN MERGING: DISABLED (quality takes priority over speed)
#    Even conservative 25% reduction still degraded quality
#    Disabling restores full quality at cost of 3-4s speedup
#    Still benefits from Flash Attention + FP8 (2.5x faster baseline)
#    Expected: 13s per video to ~11s (still 2s faster, perfect quality)
export ENABLE_TOKEN_MERGING=0

# ⚡ FP8 ENCODER/DECODER: 1.3-1.5x speedup on Conv2d layers (11-16% pipeline speedup!)
#    Uses FP8Conv2d for all encoder/decoder convolutions (Ada 4th Gen Tensor Cores)
export ENABLE_FP8_ENCODER=1
export ENABLE_FP8_DECODER=1

# ⚡ FP8 RFCNET: 1.3-1.5x speedup on Conv2d/Conv3d layers (4-7% pipeline speedup!)
#    Uses FP8Conv2d/FP8Conv3d for flow completion network (Ada 4th Gen Tensor Cores)
export ENABLE_FP8_RFCNET=1

# ⚡ DCNv4 RFCNET: 3x speedup on deformable convolution (12-18% pipeline speedup!)
#    Uses DCNv4 module with optimized CUDA kernels (replaces torchvision deform_conv2d)
#    Phase 1A: PyTorch validation (3x speedup expected)
#    Phase 2-4: TensorRT plugin (6.5-9x total speedup when combined with FP8)
export ENABLE_DCNV4_RFCNET=1

# ❌ NVDEC VIDEO DECODER: DISABLED (7.4x SLOWER than CPU due to GPU→CPU transfer overhead)
#    GPU decode is fast, but PCIe copy to CPU numpy arrays kills all gains
#    Only beneficial if entire pipeline stays on GPU (not this use case)
export ENABLE_NVDEC=0

# ⚡ TORCH COMPILE: ENABLED (1.5-2x speedup on RAFT optical flow!)
#    Sequential mode (1 worker) = no cache conflicts
#    Compiles RAFT on first run (~20-40s warmup)
#    Subsequent runs: instant cached compilation + 1.5-2x speedup
export USE_TORCH_COMPILE=1
export USE_TORCH_COMPILE_RAFT=1
export TORCH_CUDAGRAPHS=0
export TORCHINDUCTOR_CUDAGRAPHS=0

# ⚡ SEGMENT_WORKERS: Number of parallel ProPainter segment workers (must match --concurrency!)
#    Parallel mode: 4 workers (per-PID cache isolation prevents conflicts)
export SEGMENT_WORKERS=4

# ⚡ SAM2 TRACKING: Use SAM2-Tiny for temporal mask tracking (BEST QUALITY!)
#    YOLO detects bbox on first frame → SAM2 tracks with temporal consistency
#    44ms/frame, no flickering, perfect for moving watermarks!
# TEMPORARILY DISABLED - using YOLO-only for speed testing
export USE_SAM2_TRACKING=0

echo
echo "============================================================"
echo "OPTIMIZED CONFIG (RTX 4090):"
echo "  - YOLO: TensorRT batch 64 (FASTEST! 748 fps benchmark)"
echo "  - SAM2-Tiny: DISABLED (YOLO-only mode for speed)"
echo "  - Video Decode: CPU cv2.VideoCapture (7.4x faster than NVDEC for CPU pipeline!)"
echo "  - Optical Flow: NeuFlow v2 TensorRT FP16 (3-4x faster than ONNX, 10-70x faster than RAFT!)"
echo "  - RFCNet: TensorRT FP16 + DCNv4 plugin (1.6-2.3x faster flow completion!)"
echo "  - Transformer: Manual attention (FASTEST! 0.99-1.02s feature propagation!)"
echo "  - SageAttention: DISABLED (incompatible with sparse transformers!)"
echo "  - Flash Attention: DISABLED (manual attention is 5x faster!)"
echo "  - FP8 Transformer: ENABLED (1.3-1.5x speedup on Linear layers!)"
echo "  - Token Merging: DISABLED (quality priority)"
echo "  - torch.compile: ENABLED (sequential = safe, 1.5-2x RAFT speedup!)"
echo "  - FP8 Encoder/Decoder: ENABLED (1.3-1.5x speedup!)"
echo "  - FP8 RFCNet: ENABLED (1.3-1.5x speedup, 4-7% pipeline gain!)"
echo "  - DCNv4: ENABLED (3x faster deformable convolution!)"
echo "  - Concurrency: solo (SINGLE WORKER - last segment triggers finalize!)"
echo "  - SEGMENT_WORKERS: 4 (segments processed sequentially)"
echo "  - torch.compile: ENABLED (1.5-2x speedup on RAFT!)"
echo "============================================================"
echo

# ✅ ENABLE torch.compile caching (Linux atomic rename = NO FileExistsError!)
#    Per-PID cache isolation (cache_config.py) prevents multi-worker conflicts
#    Linux filesystem supports atomic rename, no race conditions!
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

echo
echo "============================================================"
echo "STARTING CELERY WORKER (WSL2 - Linux atomic rename!)..."
echo "============================================================"
echo "[DEBUG] Connecting to Redis broker: $REDIS_URL"
echo

echo "[TEST] Testing Redis connectivity..."
echo "[DEBUG] Bash script REDIS_URL: $REDIS_URL"

echo -n "[TEST] Attempting connection... "
python3 -c "import redis, sys; url = '$REDIS_URL'; r = redis.from_url(url, socket_connect_timeout=3, socket_timeout=3); r.ping(); print('[OK] Connection successful!'); sys.exit(0)"

if [ $? -ne 0 ]; then
    echo "FAILED"
    echo
    echo "[ERROR] ❌ Cannot connect to Redis server!"
    echo "[ERROR] Redis URL attempted: $REDIS_URL"
    echo
    echo "[FIX] Options:"
    echo "  1. Check redis_url.txt has correct Railway Redis URL"
    echo "  2. Verify Railway Redis service is running"
    echo "  3. Check network/firewall allows connection to Railway"
    echo "  4. For local dev: Start Redis locally with: redis-server"
    echo
    exit 1
fi

echo "[INFO] This may take 10-30s for model loading..."
echo

python3 -m celery -A server_production.celery worker --loglevel=info --pool=solo
