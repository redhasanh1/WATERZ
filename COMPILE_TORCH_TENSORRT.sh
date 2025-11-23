#!/bin/bash
# Torch-TensorRT AOT Model Compilation Script (WSL2/Linux)

echo "================================================================================"
echo "TORCH-TENSORRT AOT MODEL COMPILATION (WSL2/Linux)"
echo "================================================================================"
echo ""
echo "This will pre-compile RAFT and RFCNet models with torch-tensorrt."
echo "Compilation takes ~60-120 seconds."
echo ""
echo "Benefits:"
echo "  - Workers load models in 0.5s (not 30-60s per worker)"
echo "  - Enables true 4x parallel speedup"
echo "  - One-time compilation, reuse forever"
echo ""

cd "$(dirname "$0")"

# Check if torch-tensorrt is installed
python3 -c "import torch_tensorrt" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERROR] torch-tensorrt is not installed!"
    echo ""
    echo "Install with:"
    echo "  pip3 install torch-tensorrt --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    exit 1
fi

echo "[OK] torch-tensorrt is installed"
echo ""

# Configuration
export TORCH_TENSORRT_PRECISION=fp16
export TORCH_TENSORRT_WORKSPACE=4096

echo "[CONFIG] Precision: $TORCH_TENSORRT_PRECISION"
echo "[CONFIG] Workspace: $TORCH_TENSORRT_WORKSPACE MB"
echo ""

# Run compilation script
python3 build_torch_tensorrt_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Compilation failed! Check errors above."
    exit 1
fi

echo ""
echo "================================================================================"
echo "SUCCESS! Models compiled successfully."
echo "================================================================================"
echo ""
echo "To use compiled models:"
echo "  1. Set: export USE_TORCH_TENSORRT=1"
echo "  2. Run your script normally"
echo ""
echo "For parallel processing:"
echo "  export MAX_PARALLEL_STREAMS=4"
echo "  export PARALLEL_MODE=multiprocessing"
echo "  export USE_TORCH_TENSORRT=1"
echo ""
