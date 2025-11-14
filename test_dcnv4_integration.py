"""
Test DCNv4 integration for ProPainter BidirectionalPropagation.

This script verifies:
1. DCNv4 imports correctly
2. DCNv4 forward pass works on RTX 4090
3. Performance comparison: DCNv4 vs DCNv2 (should be 3x faster)
"""

import torch
import time
import sys

# Add DCNv4 to path (compiled but not installed as package)
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')

print("=" * 80)
print("DCNv4 INTEGRATION TEST - RTX 4090")
print("=" * 80)
print()

# Test 1: Import DCNv4
print("[1/4] Testing DCNv4 import...")
try:
    from DCNv4 import DCNv4
    print("[OK] DCNv4 imported successfully!")
except ImportError as e:
    print(f"[ERROR] Failed to import DCNv4: {e}")
    print("Run BUILD_DCNV4.bat first to install DCNv4")
    sys.exit(1)
print()

# Test 2: CUDA availability
print("[2/4] Verifying CUDA...")
if not torch.cuda.is_available():
    print("[ERROR] CUDA not available!")
    sys.exit(1)
print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
print(f"[OK] CUDA version: {torch.version.cuda}")
print()

# Test 3: DCNv4 forward pass
print("[3/4] Testing DCNv4 forward pass...")
device = torch.device('cuda')

# Create DCNv4 layer matching ProPainter usage
# BidirectionalPropagation uses: channels=256, kernel_size=3, stride=1, padding=1
try:
    dcnv4_layer = DCNv4(
        channels=256,
        kernel_size=3,
        stride=1,
        pad=1,
        group=8,  # DCNv4 optimized group size
    ).to(device)

    # Test input: [B, C, H, W] typical ProPainter features
    test_input_nchw = torch.randn(1, 256, 64, 64, device=device)

    # DCNv4 expects [N, L, C] where L=H*W
    B, C, H, W = test_input_nchw.shape
    test_input = test_input_nchw.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [1, 4096, 256]

    # Warmup
    with torch.no_grad():
        _ = dcnv4_layer(test_input, shape=(H, W))

    # Timed inference
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = dcnv4_layer(test_input, shape=(H, W))
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"[OK] DCNv4 forward pass successful!")
    print(f"[OK] Input shape (NCHW): {test_input_nchw.shape} -> (NLC): {test_input.shape}")
    print(f"[OK] Output shape: {output.shape}")
    print(f"[PERF] Inference time: {elapsed*1000:.2f}ms")
    print()

except Exception as e:
    print(f"[ERROR] DCNv4 forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Compare with DCNv2 (if available)
print("[4/4] Comparing DCNv4 vs DCNv2 performance...")
try:
    from model.modules.deform_conv import ModulatedDeformConvPack as DCNv2

    # Create DCNv2 layer with same config
    dcnv2_layer = DCNv2(
        256, 256,
        kernel_size=3,
        stride=1,
        padding=1,
        deformable_groups=8
    ).to(device)

    # Warmup (DCNv2 uses NCHW format)
    with torch.no_grad():
        _ = dcnv2_layer(test_input_nchw)

    # Timed inference
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = dcnv2_layer(test_input_nchw)
    torch.cuda.synchronize()
    dcnv2_time = time.time() - start

    speedup = dcnv2_time / elapsed
    print(f"[OK] DCNv2 inference time: {dcnv2_time*1000:.2f}ms")
    print(f"[OK] DCNv4 speedup: {speedup:.2f}x faster than DCNv2")

    if speedup >= 2.5:
        print("[SUCCESS] DCNv4 is significantly faster! (Expected: 3x)")
    elif speedup >= 1.5:
        print("[OK] DCNv4 is faster, but less than expected (check GPU utilization)")
    else:
        print("[WARNING] DCNv4 speedup lower than expected (target: 3x)")

except ImportError:
    print("[INFO] DCNv2 not found, skipping comparison")
    print("[INFO] DCNv4 is ready to integrate!")

print()
print("=" * 80)
print("[SUCCESS] DCNv4 is ready for ProPainter integration!")
print("=" * 80)
print()
print("Next step: Integrate DCNv4 into BidirectionalPropagation module")
print("  - File: faster-propainter-main/model/modules/deform_conv.py")
print("  - Replace DCNv2 imports with DCNv4 in backward propagation")
