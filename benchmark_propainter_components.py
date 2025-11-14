"""
Benchmark ProPainter components using the actual working pipeline
Measures time for feature propagation vs transformers using real video processing
"""
import os
import sys
import time
import torch

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Setup paths and environment
sys.path.insert(0, 'python_packages')
sys.path.insert(0, 'faster-propainter-main')
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')

os.environ['USE_NEUFLOW'] = '1'
os.environ['ENABLE_FLASH_ATTENTION'] = '1'

print("="*80)
print("PROPAINTER COMPONENT BENCHMARK")
print("Using real video processing pipeline with timing instrumentation")
print("="*80)
print()

# Patch ProPainter with timing instrumentation
import model.propainter as propainter_module

# Store original forward
original_bidirectional_forward = propainter_module.BidirectionalPropagation.forward

# Timing storage
timing_data = {
    'feature_propagation': [],
    'transformer': [],
    'encoder': [],
    'decoder': []
}

def timed_bidirectional_forward(self, *args, **kwargs):
    """Wrapper to time BidirectionalPropagation"""
    torch.cuda.synchronize()
    start = time.time()
    result = original_bidirectional_forward(self, *args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    timing_data['feature_propagation'].append(elapsed)
    return result

# Patch the forward method
propainter_module.BidirectionalPropagation.forward = timed_bidirectional_forward

# Patch transformer
try:
    from model.modules.sparse_transformer import SparseTransformerModule
    original_transformer_forward = SparseTransformerModule.forward

    def timed_transformer_forward(self, *args, **kwargs):
        """Wrapper to time Transformer"""
        torch.cuda.synchronize()
        start = time.time()
        result = original_transformer_forward(self, *args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        timing_data['transformer'].append(elapsed)
        return result

    SparseTransformerModule.forward = timed_transformer_forward
    print("[OK] Instrumented BidirectionalPropagation and Transformer\n")
except ImportError as e:
    print(f"[WARNING] Could not instrument transformer: {e}\n")

# Check DCNv4
try:
    from DCNv4 import DCNv4
    print("[OK] DCNv4 available (3x faster)\n")
except ImportError:
    print("[INFO] DCNv4 not available\n")

# Now import and run the pipeline
from watermark import pipeline as propainter_pipeline
import cv2
import numpy as np

# Configuration - use a short test video
test_video = 'videostotrain/first.mp4'
output_dir = 'results/benchmark_test'

if not os.path.exists(test_video):
    print(f"[ERROR] Test video not found: {test_video}")
    print("Please provide a test video or update the path")
    exit(1)

os.makedirs(output_dir, exist_ok=True)

# Load video frames (limit to 30 frames for quick benchmark)
cap = cv2.VideoCapture(test_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
frame_limit = 30  # Benchmark on 30 frames only
for i in range(frame_limit):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f"Test Configuration:")
print(f"  Video: {test_video}")
print(f"  Frames: {len(frames)}")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print()

# Create dummy masks (20% masked area in center)
masks = []
for frame in frames:
    mask = np.zeros((height, width), dtype=np.uint8)
    h_start, h_end = height//3, 2*height//3
    w_start, w_end = width//3, 2*width//3
    mask[h_start:h_end, w_start:w_end] = 255
    masks.append(mask)

# Make contiguous arrays
frames_array = [np.ascontiguousarray(f) for f in frames]
masks_array = [np.ascontiguousarray(m) for m in masks]

print("="*80)
print("RUNNING PROPAINTER PIPELINE")
print("="*80)
print()

# Run pipeline with instrumentation
start_total = time.time()
try:
    propainter_pipeline(
        video=test_video,
        mask='dummy',
        output=output_dir,
        resize_ratio=1.0,
        mask_dilation=4,
        ref_stride=15,
        neighbor_length=10,
        subvideo_length=120,
        raft_iter=10,
        mode="video_inpainting",
        save_fps=fps,
        save_frames=True,
        fp16=True,
        use_cached_models=True,
        frames_array=frames_array,
        masks_array=masks_array
    )
except Exception as e:
    print(f"[ERROR] Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

total_time = time.time() - start_total

print()
print("="*80)
print("BENCHMARK RESULTS")
print("="*80)
print()

# Calculate statistics
feature_prop_total = sum(timing_data['feature_propagation'])
transformer_total = sum(timing_data['transformer'])
encoder_total = sum(timing_data['encoder'])
decoder_total = sum(timing_data['decoder'])

feature_prop_calls = len(timing_data['feature_propagation'])
transformer_calls = len(timing_data['transformer'])

print(f"Total pipeline time: {total_time:.2f}s")
print(f"Frames processed: {len(frames)}")
print(f"Per-frame time: {(total_time/len(frames))*1000:.1f}ms")
print()

print(f"Component Breakdown:")
print(f"  Feature Propagation (DCNv4):")
print(f"    Total: {feature_prop_total:.3f}s ({feature_prop_calls} calls)")
print(f"    Average per call: {(feature_prop_total/feature_prop_calls)*1000:.1f}ms" if feature_prop_calls > 0 else "    N/A")
print(f"    % of total: {(feature_prop_total/total_time*100):.1f}%")
print()

print(f"  Transformer (Sparse Attention):")
print(f"    Total: {transformer_total:.3f}s ({transformer_calls} calls)")
print(f"    Average per call: {(transformer_total/transformer_calls)*1000:.1f}ms" if transformer_calls > 0 else "    N/A")
print(f"    % of total: {(transformer_total/total_time*100):.1f}%")
print()

categorized_time = feature_prop_total + transformer_total
other_time = total_time - categorized_time
print(f"  Other (Encoder/Decoder/Flow/etc):")
print(f"    Total: {other_time:.3f}s")
print(f"    % of total: {(other_time/total_time*100):.1f}%")
print()

# Identify bottleneck
components = [
    ("Feature Propagation", feature_prop_total),
    ("Transformer", transformer_total),
    ("Other", other_time)
]
components.sort(key=lambda x: x[1], reverse=True)

print("="*80)
print("BOTTLENECK ANALYSIS")
print("="*80)
print()

for name, time_s in components:
    pct = (time_s / total_time * 100) if total_time > 0 else 0
    bar = '█' * int(pct/2)
    print(f"{name:30s}: {time_s:6.2f}s ({pct:5.1f}%) {bar}")

print()
bottleneck_name, bottleneck_time = components[0]
print(f"Primary Bottleneck: {bottleneck_name} ({(bottleneck_time/total_time*100):.1f}% of time)")
print()

if bottleneck_name == "Feature Propagation":
    print("✓ DCNv4 already enabled (3x faster than DCNv2)")
    print("  Next optimization options:")
    print("  1. Quantize backbone Conv2d to FP8 (PyTorch native)")
    print("  2. Use Transformer Engine FP8 on linear layers")
    print("  3. Reduce deform_groups (quality tradeoff)")
elif bottleneck_name == "Transformer":
    print("  Optimization options:")
    print("  1. Use Transformer Engine FP8 quantization")
    print("  2. Reduce transformer depths [3,3,3] -> [2,2,2]")
    print("  3. Export to TensorRT with FP8")
else:
    print("  Optimization focused on encoder/decoder/flow processing")

print()
print("="*80)
print("BENCHMARK COMPLETE")
print("="*80)
