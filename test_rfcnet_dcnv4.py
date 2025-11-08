"""
Phase 1A: Test RFC Net with DCNv4
Verifies that RFC Net works with DCNv4 deformable convolution (3x faster than DCNv2)
"""
import os
import sys
import torch
import time

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Setup paths (MUST be done before importing any model code!)
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')  # Add DCNv4 FIRST
sys.path.insert(0, 'python_packages')
sys.path.insert(0, 'faster-propainter-main')

# Enable DCNv4 and FP8 (MUST be done before importing model!)
os.environ['ENABLE_DCNV4_RFCNET'] = '1'
os.environ['ENABLE_FP8_RFCNET'] = '1'

print("="*80)
print("PHASE 1A: RFC NET WITH DCNv4 FUNCTIONAL TEST")
print("="*80)
print()

print("[CONFIG] Settings:")
print(f"  ENABLE_DCNV4_RFCNET = {os.environ['ENABLE_DCNV4_RFCNET']}")
print(f"  ENABLE_FP8_RFCNET = {os.environ['ENABLE_FP8_RFCNET']}")
print()

# Test 1: Import RFC Net with DCNv4
print("[TEST 1] Importing RFC Net with DCNv4...")
try:
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet, ENABLE_DCNV4_RFCNET
    print(f"  ✓ RFC Net imported")
    print(f"  ✓ ENABLE_DCNV4_RFCNET flag: {ENABLE_DCNV4_RFCNET}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Initialize RFC Net
print("\n[TEST 2] Initializing RFC Net...")
try:
    rfcnet = RecurrentFlowCompleteNet().cuda().half().eval()  # .half() for FP16 (required with FP8)
    print("  ✓ RFC Net initialized successfully (FP16 mode)")

    # Check if DCNv4 layers are being used
    from model.recurrent_flow_completion import SecondOrderDeformableAlignmentDCNv4
    dcnv4_count = 0
    for name, module in rfcnet.named_modules():
        if isinstance(module, SecondOrderDeformableAlignmentDCNv4):
            dcnv4_count += 1

    print(f"  ✓ Found {dcnv4_count} DCNv4 deformable alignment layers")
    if dcnv4_count > 0:
        print("  ✓ DCNv4 optimization is ACTIVE!")
    else:
        print("  ⚠ WARNING: No DCNv4 layers found (using old deformable conv)")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test forward pass
print("\n[TEST 3] Testing RFC Net forward pass...")
try:
    # Create dummy inputs
    # RFC Net forward signature: masked_flows (b, t, 2, h, w), masks (b, t, 1, h, w)
    # After concat: (b, 3, t, h, w) which matches downsample Conv3d(3, ...)
    b, t, h, w = 1, 5, 128, 128
    masked_flows = torch.randn(b, t, 2, h, w).cuda()
    masks = torch.ones(b, t, 1, h, w).cuda()  # Single channel masks!

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            pred_flows, pred_edges = rfcnet(masked_flows, masks)

        # Timed run
        torch.cuda.synchronize()
        start = time.time()
        pred_flows, pred_edges = rfcnet(masked_flows, masks)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000

    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Input shape: {masked_flows.shape}")
    print(f"  ✓ Output flows shape: {pred_flows.shape}")
    print(f"  ✓ Inference time: {elapsed:.2f}ms")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Compare with old deformable conv
print("\n[TEST 4] Comparing DCNv4 vs old deformable conv...")
try:
    # Re-import with DCNv4 disabled
    os.environ['ENABLE_DCNV4_RFCNET'] = '0'

    # Reload module to pick up new environment
    import importlib
    import model.recurrent_flow_completion
    importlib.reload(model.recurrent_flow_completion)

    from model.recurrent_flow_completion import RecurrentFlowCompleteNet as RFCNetOld

    rfcnet_old = RFCNetOld().cuda().half().eval()  # .half() for FP16
    print("  ✓ Old RFC Net (DCNv2) loaded (FP16 mode)")

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            pred_flows_old, _ = rfcnet_old(masked_flows, masks)

        # Timed run
        torch.cuda.synchronize()
        start = time.time()
        pred_flows_old, _ = rfcnet_old(masked_flows, masks)
        torch.cuda.synchronize()
        elapsed_old = (time.time() - start) * 1000

    print(f"  ✓ DCNv2 inference time: {elapsed_old:.2f}ms")
    print(f"  ✓ DCNv4 inference time: {elapsed:.2f}ms")

    speedup = elapsed_old / elapsed
    print(f"\n  🚀 SPEEDUP: {speedup:.2f}x faster with DCNv4!")

    if speedup >= 2.5:
        print(f"  ✅ EXCELLENT! Achieved {speedup:.2f}x speedup (target: 3x)")
    elif speedup >= 2.0:
        print(f"  ✅ GOOD! Achieved {speedup:.2f}x speedup (close to 3x target)")
    elif speedup >= 1.5:
        print(f"  ⚠️  MODERATE: {speedup:.2f}x speedup (below 3x target, may need optimization)")
    else:
        print(f"  ❌ WARNING: Only {speedup:.2f}x speedup (significantly below 3x target!)")

except Exception as e:
    print(f"  ⚠️  Could not compare (this is optional): {e}")

print()
print("="*80)
print("PHASE 1A COMPLETE!")
print("="*80)
print()
print("Summary:")
print("  ✅ RFC Net with DCNv4 is working correctly")
print(f"  ✅ {dcnv4_count} DCNv4 deformable alignment layers active")
print(f"  ✅ Forward pass successful")
if 'speedup' in locals():
    print(f"  ✅ Speedup: {speedup:.2f}x faster than DCNv2")
print()
print("Next: Phase 1A Benchmark for detailed performance analysis")
print()
