#!/bin/bash
# ============================================================
# Build YOLO TensorRT Engine for WSL2/Linux
# ============================================================
# This creates a TensorRT engine optimized for YOLO detection
# in WSL, making the sender ~20-30x faster than PyTorch.
#
# Prerequisites:
#   - CUDA installed in WSL
#   - TensorRT Python package: pip install tensorrt
#   - PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121
#
# Usage:
#   cd /mnt/d/watermarkz
#   chmod +x build_yolo_trt_wsl.sh
#   ./build_yolo_trt_wsl.sh
# ============================================================

echo ""
echo "============================================================"
echo "YOLO TensorRT Engine Builder (WSL)"
echo "============================================================"
echo ""

cd /mnt/d/watermarkz

# Activate virtual environment
if [ -d "venv_wsl2" ]; then
    source venv_wsl2/bin/activate
    echo "[OK] Activated venv_wsl2"
else
    echo "[ERROR] venv_wsl2 not found!"
    echo "Create it with: python3 -m venv venv_wsl2"
    exit 1
fi

# Check TensorRT is installed
python -c "import tensorrt" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] TensorRT not installed!"
    echo "Install with: pip install tensorrt"
    exit 1
fi

# Check CUDA is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] PyTorch CUDA not available!"
    echo "Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

echo "[OK] TensorRT and CUDA available"
echo ""

# Run the builder
python build_yolo_trt_engine_wsl.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "[SUCCESS] Engine built!"
    echo ""
    echo "The sender will now use TensorRT for YOLO detection."
    echo "Expected speedup: 20-30x faster than PyTorch"
    echo "============================================================"
else
    echo ""
    echo "[FAILED] Engine build failed - check errors above"
    exit 1
fi
