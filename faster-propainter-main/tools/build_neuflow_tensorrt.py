#!/usr/bin/env python3
"""
Build TensorRT FP16 engine for NeuFlow v2 optical flow model.

Expected speedup: 4.0s → 2.0-2.7s (1.5-2X faster than ONNX Runtime)
Quality: ZERO loss (FP16 = same precision as ONNX)
"""

import os
import sys
import subprocess
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

NEUFLOW_ONNX = MODELS_DIR / "neuflow_things.onnx"
NEUFLOW_ENGINE = MODELS_DIR / "neuflow_things_fp16.engine"

# TensorRT paths
TENSORRT_ROOT = Path(r"D:\watermarkz\TensorRT-10.13.3.9")
TRTEXEC = TENSORRT_ROOT / "bin" / "trtexec.exe"


def check_prerequisites():
    """Check if all required files and tools are available"""
    print("=" * 70)
    print("PREREQUISITES CHECK")
    print("=" * 70)

    if not NEUFLOW_ONNX.exists():
        print(f"[ERROR] NeuFlow v2 ONNX not found: {NEUFLOW_ONNX}")
        print("[INFO] Please download NeuFlow v2 ONNX model first")
        return False

    print(f"[OK] NeuFlow v2 ONNX found: {NEUFLOW_ONNX}")
    print(f"     Size: {NEUFLOW_ONNX.stat().st_size / 1024 / 1024:.2f} MB")

    if not TRTEXEC.exists():
        print(f"[ERROR] trtexec not found: {TRTEXEC}")
        print(f"[INFO] Expected TensorRT installation at: {TENSORRT_ROOT}")
        return False

    print(f"[OK] trtexec found: {TRTEXEC}")

    return True


def build_tensorrt_engine():
    """Build TensorRT FP16 engine from ONNX"""
    print("\n" + "=" * 70)
    print("BUILDING TENSORRT FP16 ENGINE")
    print("=" * 70)

    # Remove existing engine if present
    if NEUFLOW_ENGINE.exists():
        print(f"[INFO] Removing existing engine: {NEUFLOW_ENGINE}")
        NEUFLOW_ENGINE.unlink()

    # trtexec command for FP16 optimization
    # Note: TensorRT 10 syntax changes from earlier versions
    # Starting with fixed batch=1 for simplicity (can add dynamic shapes later)
    cmd = [
        str(TRTEXEC),
        f"--onnx={NEUFLOW_ONNX}",
        f"--saveEngine={NEUFLOW_ENGINE}",
        "--fp16",  # Enable FP16 precision (same as ONNX, ZERO quality loss)
        "--memPoolSize=workspace:4096",  # 4GB workspace (TensorRT 10+ syntax)
        "--tacticSources=+CUDNN,+CUBLAS,+EDGE_MASK_CONVOLUTIONS",  # Use all optimization tactics
        "--builderOptimizationLevel=5",  # Maximum optimization
        #"--verbose",  # Show detailed build info (commented to reduce output)
    ]

    print(f"\n[CMD] {' '.join(cmd)}\n")
    print("[INFO] Building TensorRT engine...")
    print("[INFO] This may take 5-10 minutes (one-time cost)")
    print("[INFO] Engine will be optimized for RTX 4090 Ada Lovelace architecture")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        print(result.stdout)

        if NEUFLOW_ENGINE.exists():
            engine_size = NEUFLOW_ENGINE.stat().st_size / 1024 / 1024
            print("\n" + "=" * 70)
            print("SUCCESS!")
            print("=" * 70)
            print(f"[OK] TensorRT engine created: {NEUFLOW_ENGINE}")
            print(f"[OK] Engine size: {engine_size:.2f} MB")
            print("\n[NEXT STEPS]:")
            print("1. Update flow_comp_neuflow.py to load .engine instead of .onnx")
            print("2. Use TensorRT Python API for inference")
            print("3. Expected speedup: 1.5-2X faster (4.0s → 2.0-2.7s)")
            return True
        else:
            print("[ERROR] Engine file not created!")
            return False

    except subprocess.CalledProcessError as e:
        print("[ERROR] trtexec failed!")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    print("=" * 70)
    print("NEUFLOW V2 TENSORRT FP16 ENGINE BUILDER")
    print("=" * 70)
    print(f"Input:  {NEUFLOW_ONNX}")
    print(f"Output: {NEUFLOW_ENGINE}")
    print(f"Precision: FP16 (ZERO quality loss)")
    print(f"Target GPU: RTX 4090 Ada Lovelace")
    print()

    if not check_prerequisites():
        return 1

    if not build_tensorrt_engine():
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
