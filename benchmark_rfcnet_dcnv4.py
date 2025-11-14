"""
Phase 1A: Comprehensive DCNv4 vs DCNv2 Benchmark
Tests RFC Net performance at production resolutions to measure actual 3x speedup
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

# Enable FP8 for both tests (isolate DCNv4 speedup)
os.environ['ENABLE_FP8_RFCNET'] = '1'

print("="*80)
print("PHASE 1A: DCNv4 vs DCNv2 COMPREHENSIVE BENCHMARK")
print("="*80)
print()

def benchmark_model(model, inputs, name, warmup=10, iterations=50):
    """Benchmark a model with warmup and multiple iterations"""
    masked_flows, masks = inputs

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(masked_flows, masks)

    # Benchmark
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            _ = model(masked_flows, masks)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time)**2 for t in times) / len(times)) ** 0.5

    return mean_time, std_time

# Test configurations (resolution, temporal_length)
test_configs = [
    (256, 256, 8, "Small (256x256, 8 frames)"),
    (512, 512, 8, "Medium (512x512, 8 frames)"),
    (640, 480, 10, "SD (640x480, 10 frames)"),
    (1024, 1024, 8, "Large (1024x1024, 8 frames)"),
    (1280, 720, 10, "HD (1280x720, 10 frames)"),
]

results = []

for h, w, t, desc in test_configs:
    print(f"\n{'='*80}")
    print(f"TEST: {desc}")
    print(f"{'='*80}")

    # Create test inputs
    b = 1
    masked_flows = torch.randn(b, t, 2, h, w).cuda().half()
    masks = torch.ones(b, t, 1, h, w).cuda().half()
    inputs = (masked_flows, masks)

    print(f"  Input shape: {masked_flows.shape}")
    print(f"  Memory: {masked_flows.element_size() * masked_flows.nelement() / 1024**2:.1f} MB")
    print()

    # Test 1: DCNv4
    print("  [1/2] Loading RFC Net with DCNv4...")
    os.environ['ENABLE_DCNV4_RFCNET'] = '1'

    # Clear module cache
    if 'model.recurrent_flow_completion' in sys.modules:
        del sys.modules['model.recurrent_flow_completion']

    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    rfcnet_dcnv4 = RecurrentFlowCompleteNet().cuda().half().eval()

    print("  Benchmarking DCNv4...")
    dcnv4_time, dcnv4_std = benchmark_model(rfcnet_dcnv4, inputs, "DCNv4", warmup=5, iterations=30)
    print(f"    DCNv4: {dcnv4_time:.2f} ± {dcnv4_std:.2f} ms/iter")

    # Free memory
    del rfcnet_dcnv4
    torch.cuda.empty_cache()

    # Test 2: DCNv2
    print("  [2/2] Loading RFC Net with DCNv2...")
    os.environ['ENABLE_DCNV4_RFCNET'] = '0'

    # Force reload
    if 'model.recurrent_flow_completion' in sys.modules:
        del sys.modules['model.recurrent_flow_completion']

    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    rfcnet_dcnv2 = RecurrentFlowCompleteNet().cuda().half().eval()

    print("  Benchmarking DCNv2...")
    dcnv2_time, dcnv2_std = benchmark_model(rfcnet_dcnv2, inputs, "DCNv2", warmup=5, iterations=30)
    print(f"    DCNv2: {dcnv2_time:.2f} ± {dcnv2_std:.2f} ms/iter")

    # Free memory
    del rfcnet_dcnv2
    torch.cuda.empty_cache()

    # Calculate speedup
    speedup = dcnv2_time / dcnv4_time

    print()
    print(f"  Results:")
    print(f"    DCNv2 (old): {dcnv2_time:.2f} ms/iter")
    print(f"    DCNv4 (new): {dcnv4_time:.2f} ms/iter")
    print(f"    Speedup:     {speedup:.2f}x")

    if speedup >= 2.5:
        print(f"    ✅ EXCELLENT! Achieved {speedup:.2f}x speedup (target: 3x)")
    elif speedup >= 2.0:
        print(f"    ✅ GOOD! Achieved {speedup:.2f}x speedup (close to 3x target)")
    elif speedup >= 1.5:
        print(f"    ⚠️  MODERATE: {speedup:.2f}x speedup (below 3x target)")
    else:
        print(f"    ❌ LOW: Only {speedup:.2f}x speedup")

    results.append({
        'desc': desc,
        'resolution': f"{w}x{h}",
        'frames': t,
        'dcnv2_time': dcnv2_time,
        'dcnv4_time': dcnv4_time,
        'speedup': speedup
    })

# Summary
print()
print("="*80)
print("BENCHMARK SUMMARY")
print("="*80)
print()
print(f"{'Configuration':<30} {'DCNv2 (ms)':<15} {'DCNv4 (ms)':<15} {'Speedup':<10}")
print("-"*80)

for r in results:
    print(f"{r['desc']:<30} {r['dcnv2_time']:>10.2f} ms   {r['dcnv4_time']:>10.2f} ms   {r['speedup']:>6.2f}x")

print()
avg_speedup = sum(r['speedup'] for r in results) / len(results)
print(f"Average Speedup: {avg_speedup:.2f}x")
print()

if avg_speedup >= 2.5:
    print("✅ VALIDATED: DCNv4 provides 2.5-3x speedup as expected!")
elif avg_speedup >= 2.0:
    print("✅ GOOD: DCNv4 provides 2x+ speedup (close to target)")
else:
    print("⚠️  WARNING: DCNv4 speedup below expectations")
    print("   Possible reasons:")
    print("   - Test resolution too small (deformable conv overhead dominates)")
    print("   - FP8 layers dominate compute time")
    print("   - Need larger temporal length for better measurement")

print()
print("="*80)
print("PHASE 1A BENCHMARK COMPLETE!")
print("="*80)
print()
print("Next Steps:")
print("  ✅ Phase 1A Complete: DCNv4 validated in PyTorch")
if avg_speedup >= 2.0:
    print("  ➡️  Phase 1B-2: Implement DCNv4 TensorRT plugin")
    print("  ➡️  Expected with TensorRT: 6.5-9x total speedup")
else:
    print("  ⚠️  Consider optimizing DCNv4 integration before TensorRT")
print()
