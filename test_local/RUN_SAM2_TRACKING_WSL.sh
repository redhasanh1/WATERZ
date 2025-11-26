#!/bin/bash
# Run SAM2 tracking in WSL2 with torch.compile enabled

echo "============================================================"
echo "SAM2 TRACKING - WSL2 with torch.compile (2x faster!)"
echo "============================================================"
echo ""

cd /mnt/d/watermarkz

# WSL2/Linux: Enable torch.compile with Triton
export TORCHINDUCTOR_CACHE_DIR=/mnt/d/watermarkz/.torch_compile_cache_sam2
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0
export TRITON_CACHE_DIR=/mnt/d/watermarkz/.triton_cache
export TORCH_CUDAGRAPHS=0
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCHINDUCTOR_COMPILE_THREADS=1

echo "torch.compile ENABLED for SAM2 memory components"
echo "  - memory_attention (compiled)"
echo "  - sam_mask_decoder (compiled)"
echo "  - memory_encoder (compiled)"
echo ""
echo "Running test_sam2_tracking.py..."
echo ""

python3 test_local/test_sam2_tracking.py

echo ""
echo "Done! Now run RUN_TEST_WSL.bat for inpainting."
