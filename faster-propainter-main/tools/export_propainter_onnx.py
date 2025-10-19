"""
Skeleton exporter for the ProPainter InpaintGenerator.

The current implementation documents the required inputs and explains why
direct ONNX export fails (deformable convolution + sparse transformer blocks).
It exits with a descriptive message so users know to install/enable the
necessary TensorRT plugins or implement a mixed PyTorch/TensorRT wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent
REPO_ROOT = PACKAGE_ROOT.parent
for path in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def summarize_inputs():
    print("ProPainter InpaintGenerator expects the following inputs:")
    print("  masked_frames   : [B, T_total, 3, H, W]")
    print("  completed_flows : tuple(forward:[B, T_local-1, 2, H, W], backward:[...])")
    print("  masks_in        : [B, T_total, 1, H, W]")
    print("  masks_updated   : [B, T_local, 1, H, W]")
    print("  num_local_frames: int (typically short_clip_len)")
    print()
    print("Key blockers for ONNX export:")
    print("  - torchvision.ops.deform_conv2d (requires DCNv2 TensorRT plugin).")
    print("  - Sparse transformer blocks using fold/unfold and attention.")
    print("  - Python-level control flow around temporal padding.")
    print()
    print("Recommended next steps:")
    print("  1. Integrate DCNv2 TensorRT plugin (or keep deform alignment in PyTorch).")
    print("  2. Replace SoftSplit/SoftComp with conv-based equivalents or plugin.")
    print("  3. Fix the sequence length (short_clip_len) and mask shapes for export.")
    print("  4. Once the above are addressed, replicate the pattern from export_rfcnet_onnx.py.")


def main():
    parser = argparse.ArgumentParser(description="ProPainter ONNX exporter placeholder.")
    parser.add_argument("--weights", type=Path, default=REPO_ROOT / "web" / "weights" / "ProPainter.pth")
    parser.parse_args()

    summarize_inputs()
    print("\n[INFO] ProPainter ONNX export not yet implemented due to unsupported operators.")
    print("       See documentation in TENSORRT_OPTIMISATION.txt for details.")


if __name__ == "__main__":
    main()

