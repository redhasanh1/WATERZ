"""
Simple ProPainter Profiler using torch.profiler
Automatically identifies bottlenecks without manual instrumentation
"""
import os
import sys
import torch
import time

# Add paths
sys.path.insert(0, 'faster-propainter-main')
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')

# Enable configurations
os.environ['USE_NEUFLOW'] = '1'
os.environ['ENABLE_FLASH_ATTENTION'] = '1'

print("="*80)
print("PROPAINTER SIMPLE PROFILER")
print("="*80)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print()

# Check DCNv4
try:
    from DCNv4 import DCNv4
    print("[OK] DCNv4 enabled")
except ImportError:
    print("[INFO] DCNv4 not available")

from model.propainter import InpaintGenerator

# Configuration
frames_count = 30
h, w = 480, 640
device = torch.device('cuda')

# Load model
print("\nLoading ProPainter...")
model = InpaintGenerator(model_path='weights/ProPainter.pth').to(device)
model.eval()
print("[OK] Model loaded\n")

# Create test data
with torch.no_grad():
    frames = torch.randn(1, frames_count, 3, h, w, device=device)
    masks = (torch.rand(1, frames_count, 1, h, w, device=device) > 0.8).float()
    masked_frames = frames * (1 - masks)
    flows_f = torch.randn(1, frames_count-1, 2, h, w, device=device)
    flows_b = torch.randn(1, frames_count-1, 2, h, w, device=device)

# Warmup
print("Warmup...")
with torch.no_grad():
    _ = model(masked_frames, masks, flows_f, flows_b, 'video_completion')
torch.cuda.synchronize()
print("[OK] Warmup complete\n")

# Benchmark end-to-end first
print("="*80)
print("END-TO-END BENCHMARK")
print("="*80)

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    _ = model(masked_frames, masks, flows_f, flows_b, 'video_completion')
torch.cuda.synchronize()
elapsed = time.time() - start

per_frame_ms = (elapsed / frames_count) * 1000
print(f"Total time: {elapsed:.3f}s")
print(f"Per frame: {per_frame_ms:.1f}ms")
print(f"FPS: {frames_count/elapsed:.1f}")
print()

# Now profile with detailed breakdown
print("="*80)
print("DETAILED PROFILING (Top Operations)")
print("="*80)
print()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=False,
    with_stack=False,
) as prof:
    with torch.no_grad():
        _ = model(masked_frames, masks, flows_f, flows_b, 'video_completion')

# Print top operations by CUDA time
print("Top 20 operations by CUDA time:\n")
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20,
    top_level_events_only=False
))

# Print summary grouped by key module names
print("\n" + "="*80)
print("MODULE-LEVEL BREAKDOWN")
print("="*80)
print()

# Get events
events = prof.key_averages()

# Group by module/operation type
module_times = {}
for evt in events:
    name = evt.key
    cuda_time_ms = evt.cuda_time_total / 1000  # Convert to ms

    if cuda_time_ms < 1:  # Skip ops < 1ms
        continue

    # Categorize operations
    if 'deform' in name.lower() or 'dcn' in name.lower():
        category = "Feature Propagation (DCNv4)"
    elif 'attention' in name.lower() or 'softmax' in name.lower() or 'matmul' in name.lower():
        category = "Transformer (Attention)"
    elif 'conv' in name.lower():
        category = "Conv Operations (Encoder/Decoder)"
    elif 'interpolate' in name.lower() or 'upsample' in name.lower():
        category = "Upsampling/Interpolation"
    elif 'warp' in name.lower() or 'grid_sample' in name.lower():
        category = "Flow Warping"
    else:
        category = "Other"

    if category not in module_times:
        module_times[category] = 0
    module_times[category] += cuda_time_ms

# Print module breakdown
total_categorized = sum(module_times.values())
print(f"Module Breakdown (categorized time: {total_categorized:.1f}ms):\n")

sorted_modules = sorted(module_times.items(), key=lambda x: x[1], reverse=True)
for module_name, time_ms in sorted_modules:
    pct = (time_ms / total_categorized * 100) if total_categorized > 0 else 0
    bar = '█' * int(pct/2)
    print(f"{module_name:40s}: {time_ms:7.1f}ms ({pct:5.1f}%) {bar}")

print()
print("="*80)
print("PROFILING COMPLETE")
print("="*80)
print()

# Recommendations
print("OPTIMIZATION RECOMMENDATIONS:")
print()

if sorted_modules:
    bottleneck_name, bottleneck_time = sorted_modules[0]
    print(f"Primary Bottleneck: {bottleneck_name}")
    print()

    if "Feature Propagation" in bottleneck_name:
        print("✓ DCNv4 already enabled (3x faster than DCNv2)")
        print("  Further optimization options:")
        print("  1. Quantize backbone Conv2d layers to FP8 (PyTorch native)")
        print("  2. Use torch.compile on individual propagation steps")
        print("  3. Reduce deform_groups (quality tradeoff)")

    elif "Transformer" in bottleneck_name:
        print("  Optimization options:")
        print("  1. Use FP8 quantization via Transformer Engine")
        print("  2. Reduce transformer depths [3,3,3] -> [2,2,2]")
        print("  3. Export transformer to TensorRT with FP8")
        print("  4. Use torch.compile on transformer module")

    elif "Conv Operations" in bottleneck_name:
        print("  Optimization options:")
        print("  1. Quantize Conv2d layers to FP8/INT8")
        print("  2. Use torch.compile on encoder/decoder")
        print("  3. Export encoder/decoder to TensorRT")

print()
