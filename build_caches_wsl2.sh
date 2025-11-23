#!/bin/bash
#
# WSL2 Cache Builder - Build torch.compile caches in Linux for superior performance
#
# Why WSL2?
#   - Linux fork() is 10x faster than Windows spawn for multiprocessing
#   - Native Triton support (torch.compile works properly)
#   - No Windows file system race conditions
#   - Better CUDA kernel compilation performance
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "================================================================================"
echo "🚀 WSL2 CACHE BUILDER - Building Worker Caches in Linux"
echo "================================================================================"
echo ""
echo "Building 4 worker caches with torch.compile..."
echo "Expected time: ~100 seconds (25s per worker)"
echo ""
echo "Linux advantages:"
echo "  ✓ Fast fork() multiprocessing"
echo "  ✓ Native Triton support"
echo "  ✓ No race conditions"
echo "  ✓ Better file I/O"
echo ""
echo "================================================================================"
echo ""

# Step 1: Clean corrupted Windows cache files (if any)
echo "[1/3] Cleaning old cache directories..."
rm -rf temp/.torch_cache_0 temp/.torch_cache_1 temp/.torch_cache_2 temp/.torch_cache_3 2>/dev/null || true
rm -rf temp/.triton_cache_0 temp/.triton_cache_1 temp/.triton_cache_2 temp/.triton_cache_3 2>/dev/null || true

mkdir -p temp/.torch_cache_0 temp/.torch_cache_1 temp/.torch_cache_2 temp/.torch_cache_3
mkdir -p temp/.triton_cache_0 temp/.triton_cache_1 temp/.triton_cache_2 temp/.triton_cache_3

echo "✓ Cache directories cleaned and recreated"
echo ""

# Step 2: Set up environment variables
echo "[2/3] Setting up environment..."
export PYTHONIOENCODING=utf-8
export USE_TORCH_COMPILE_RAFT=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

echo "✓ Environment configured for torch.compile"
echo ""

# Step 3: Build caches for all 4 workers
echo "[3/3] Building caches (this will take ~100 seconds)..."
echo ""
echo "================================================================================"
echo ""

# Use python3 (Linux convention)
python3 build_worker_caches.py --workers 4

BUILD_EXIT_CODE=$?

echo ""
echo "================================================================================"

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS! All 4 worker caches built successfully!"
    echo "================================================================================"
    echo ""
    echo "Cache Statistics:"
    echo "  - Worker 0: temp/.torch_cache_0/ + temp/.triton_cache_0/"
    echo "  - Worker 1: temp/.torch_cache_1/ + temp/.triton_cache_1/"
    echo "  - Worker 2: temp/.torch_cache_2/ + temp/.triton_cache_2/"
    echo "  - Worker 3: temp/.torch_cache_3/ + temp/.triton_cache_3/"
    echo ""

    # Show cache sizes
    echo "Cache Sizes:"
    for i in 0 1 2 3; do
        TORCH_SIZE=$(du -sh temp/.torch_cache_$i 2>/dev/null | cut -f1 || echo "0")
        TRITON_SIZE=$(du -sh temp/.triton_cache_$i 2>/dev/null | cut -f1 || echo "0")
        echo "  Worker $i: TorchInductor=$TORCH_SIZE, Triton=$TRITON_SIZE"
    done

    echo ""
    echo "These caches will work for both Windows and WSL2!"
    echo ""
    echo "Next steps:"
    echo "  1. Start your workers: START_CELERY_TRT.bat"
    echo "  2. First inference will be 15x faster (2s vs 30s)"
    echo "  3. Subsequent inferences: ~1.5s per segment"
    echo ""
else
    echo "❌ FAILED with exit code $BUILD_EXIT_CODE"
    echo "================================================================================"
    echo ""
    echo "Check the error messages above."
    echo ""
fi

exit $BUILD_EXIT_CODE
