"""
load_ts_models.py - Helper to load .ts files in watermark.py

Checks for pre-compiled .ts files and loads them instantly.
Falls back to regular PyTorch if .ts files not found.

Usage in watermark.py:
    from load_ts_models import try_load_ts_model

    encoder = try_load_ts_model("raft_encoder_aot.ts", fallback_model)
"""

import os
import torch
from pathlib import Path


def try_load_ts_model(ts_filename, fallback_model=None, device='cuda'):
    """
    Try to load a .ts file, fallback to regular model if not found.

    Args:
        ts_filename: Name of .ts file (e.g. "raft_encoder_aot.ts")
        fallback_model: Model to use if .ts not found
        device: Device to load model on

    Returns:
        Loaded model (either .ts or fallback)
    """
    # Search for .ts file in multiple locations
    search_paths = [
        Path(__file__).parent.parent / "weights" / ts_filename,  # D:\watermarkz\weights\
        Path(__file__).parent / "weights" / ts_filename,         # faster-propainter-main\weights\
        Path(ts_filename),                                       # Current directory
    ]

    ts_path = None
    for path in search_paths:
        if path.exists():
            ts_path = path
            break

    if ts_path is None:
        if fallback_model is not None:
            print(f"[LOAD] .ts file not found: {ts_filename}")
            print(f"[FALLBACK] Using regular PyTorch model")
            return fallback_model
        else:
            raise FileNotFoundError(f".ts file not found: {ts_filename}")

    # Load .ts file
    print(f"[TS] Loading pre-compiled model: {ts_path.name}")
    load_start = __import__('time').time()

    try:
        # Try torch_tensorrt.load first (if available)
        try:
            import torch_tensorrt
            model = torch_tensorrt.load(str(ts_path))
            print(f"[OK] Loaded with torch_tensorrt in {__import__('time').time() - load_start:.2f}s")
        except ImportError:
            # Fallback to torch.jit.load
            model = torch.jit.load(str(ts_path), map_location=device)
            print(f"[OK] Loaded with torch.jit in {__import__('time').time() - load_start:.2f}s")

        # Move to device if needed
        if hasattr(model, 'to'):
            model = model.to(device)

        return model

    except Exception as e:
        print(f"[ERROR] Failed to load .ts file: {e}")
        if fallback_model is not None:
            print(f"[FALLBACK] Using regular PyTorch model")
            return fallback_model
        else:
            raise


def check_ts_files_available():
    """
    Check if any .ts files are available for loading.

    Returns:
        dict: {model_name: file_path} for available .ts files
    """
    ts_files = {
        'raft_encoder': 'raft_encoder_aot.ts',
        'rfcnet_downsampler': 'rfcnet_downsampler_aot.ts',
        'propainter_encoder': 'propainter_encoder_aot.ts',
    }

    available = {}

    for model_name, filename in ts_files.items():
        search_paths = [
            Path(__file__).parent.parent / "weights" / filename,
            Path(__file__).parent / "weights" / filename,
        ]

        for path in search_paths:
            if path.exists():
                available[model_name] = path
                break

    return available


if __name__ == "__main__":
    # Test: Check what .ts files are available
    print("="*80)
    print("CHECKING FOR .TS FILES")
    print("="*80)

    available = check_ts_files_available()

    if available:
        print(f"Found {len(available)} .ts files:")
        for name, path in available.items():
            size_mb = path.stat().st_size / (1024**2)
            print(f"  ✅ {name}: {path.name} ({size_mb:.1f} MB)")
    else:
        print("❌ No .ts files found")
        print()
        print("Run: python compile_hybrid_ts.py")
        print("     to build .ts files for instant loading")

    print("="*80)
