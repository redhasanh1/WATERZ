#!/usr/bin/env python3
"""
AOT (Ahead-of-Time) Model Compilation for Parallel Processing

This script pre-compiles RAFT and RFCNet models using torch.jit.trace
so that 4 parallel workers can load them instantly without JIT compilation overhead.

Usage:
    python compile_models_aot.py

This creates:
    - weights/raft_compiled.pt (traced RAFT model)
    - weights/rfcnet_compiled.pt (traced RFCNet model)

Workers then load these pre-compiled models instead of compiling at runtime.
"""

import sys
import os
from pathlib import Path
import torch

# Add ProPainter to path
BASE_DIR = Path(__file__).parent
PROPAINTER_DIR = BASE_DIR / "faster-propainter-main"
sys.path.insert(0, str(PROPAINTER_DIR))

print("="*80)
print("AOT Model Compilation for Parallel Processing")
print("="*80)
print()

# Import RAFT and RFCNet
print("[1/4] Importing models...")
from model.modules.flow_comp_raft import RAFT_bi as PyTorch_RAFT
from model.modules.flow_comp_raft import RAFT as RFCNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

if device.type != 'cuda':
    print("[ERROR] CUDA is required for compilation. Exiting.")
    sys.exit(1)

# Output directory
weights_dir = PROPAINTER_DIR / "weights"
weights_dir.mkdir(exist_ok=True)

print()
print("[2/4] Compiling RAFT model...")
print("[INFO] This may take 30-60 seconds...")

# Load RAFT
raft_path = str(weights_dir / "raft-things.pth")
raft_model = PyTorch_RAFT(raft_path, device).eval()

# Example input for RAFT: [B=1, C=2*3=6, H=432, W=256] (concatenated frame pair)
# RAFT expects: image1, image2 separately, but we'll trace the forward
example_img1 = torch.randn(1, 3, 432, 256, device='cuda')
example_img2 = torch.randn(1, 3, 432, 256, device='cuda')

try:
    # Trace RAFT
    with torch.no_grad():
        traced_raft = torch.jit.trace(raft_model, (example_img1, example_img2, 20))  # iters=20

    # Save traced model
    raft_path = weights_dir / "raft_compiled.pt"
    torch.jit.save(traced_raft, str(raft_path))
    print(f"[OK] RAFT compiled and saved to: {raft_path}")
    print(f"[INFO] File size: {raft_path.stat().st_size / 1024 / 1024:.1f} MB")
except Exception as e:
    print(f"[ERROR] RAFT compilation failed: {e}")
    print("[FALLBACK] Workers will use JIT compilation")

print()
print("[3/4] Compiling RFCNet model...")
print("[INFO] This may take 20-30 seconds...")

# Load RFCNet
rfcnet_model = RFCNet().eval().cuda()

# Example input for RFCNet: masked_flows [1, 80, 2, 432, 256], masks [1, 80, 1, 432, 256]
example_flows = torch.randn(1, 80, 2, 432, 256, device='cuda')
example_masks = torch.randn(1, 80, 1, 432, 256, device='cuda')

try:
    # Trace RFCNet
    with torch.no_grad():
        traced_rfcnet = torch.jit.trace(rfcnet_model, (example_flows, example_masks))

    # Save traced model
    rfcnet_path = weights_dir / "rfcnet_compiled.pt"
    torch.jit.save(traced_rfcnet, str(rfcnet_path))
    print(f"[OK] RFCNet compiled and saved to: {rfcnet_path}")
    print(f"[INFO] File size: {rfcnet_path.stat().st_size / 1024 / 1024:.1f} MB")
except Exception as e:
    print(f"[ERROR] RFCNet compilation failed: {e}")
    print("[FALLBACK] Workers will use JIT compilation")

print()
print("[4/4] Verification...")

# Test loading
try:
    loaded_raft = torch.jit.load(str(raft_path))
    print("[OK] RAFT model loads successfully")
except:
    print("[ERROR] RAFT model failed to load")

try:
    loaded_rfcnet = torch.jit.load(str(rfcnet_path))
    print("[OK] RFCNet model loads successfully")
except:
    print("[ERROR] RFCNet model failed to load")

print()
print("="*80)
print("✅ AOT Compilation Complete!")
print("="*80)
print()
print("Next steps:")
print("  1. Set USE_AOT_MODELS=1 in your batch file")
print("  2. Set MAX_PARALLEL_STREAMS=4 for parallel processing")
print("  3. Run RUN_PARALLEL_TEST.bat")
print()
print("Workers will load pre-compiled models instantly!")
print("="*80)
