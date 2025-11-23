#!/bin/bash
# WSL2 Dependencies Installation Script

echo "========================================="
echo "Installing WSL2 Dependencies"
echo "========================================="
echo ""

# Update system
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential git

# Install PyTorch with CUDA
echo ""
echo "[2/6] Installing PyTorch 2.5.1 with CUDA 12.4..."
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
echo ""
echo "[3/6] Installing core Python packages..."
pip3 install opencv-python numpy pillow scipy imageio tqdm scikit-image lpips tensorboard

# FFmpeg
echo ""
echo "[4/6] Installing FFmpeg..."
pip3 install static-ffmpeg

# SAM2
echo ""
echo "[5/6] Installing SAM2..."
pip3 install segment-anything-2

# SageAttention
echo ""
echo "[6/6] Installing SageAttention..."
pip3 install sageattention || echo "SageAttention install failed (optional)"

# Verify CUDA
echo ""
echo "========================================="
echo "Verifying Installation"
echo "========================================="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "========================================="
echo "✅ Installation Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. cd ~/watermarkz"
echo "  2. ./RUN_WSL2_PARALLEL.sh"
echo ""
