#!/bin/bash
# Run optimized SAM2 test in WSL2 with torch.compile enabled

echo "============================================================"
echo "OPTIMIZED SAM2 TEST - WSL2 with torch.compile"
echo "============================================================"
echo ""

# Change to watermarkz directory (mounted from Windows)
cd /mnt/d/watermarkz

# WSL2/Linux: Enable torch.compile with Triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/d/watermarkz/.torch_compile_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TRITON_CACHE_DIR=/mnt/d/watermarkz/.triton_cache
export USE_TORCH_COMPILE_RAFT=1
export TORCH_CUDAGRAPHS=0
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCHINDUCTOR_COMPILE_THREADS=1

# Production optimizations
export USE_NEUFLOW=1
export FORCE_TRT_RFCNET=1
export ENABLE_SAGE_ATTENTION=0
export ENABLE_FLASH_ATTENTION=0
export ENABLE_FP8_TRANSFORMER=1
export ENABLE_FP8_RFCNET=1
export ENABLE_DCNV4_RFCNET=1
export ENABLE_BACKGROUND_ENCODER=0
export TORCH_COMPILE_MODE=max-autotune

echo "torch.compile ENABLED (WSL2/Linux with Triton)"
echo ""
echo "Running test_sam2_optimized.py..."
echo ""

# Run the test
python3 test_local/test_sam2_optimized.py

echo ""
echo "Done!"
