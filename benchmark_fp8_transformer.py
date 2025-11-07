"""
Benchmark FP8 Transformer Performance
Compares transformer speed with FP8 ON vs FP8 OFF
"""
import os
import sys
import time
import codecs

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Setup paths
sys.path.insert(0, 'python_packages')
sys.path.insert(0, 'faster-propainter-main')
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')

print("="*80)
print("FP8 TRANSFORMER BENCHMARK")
print("Comparing FP8 ON vs FP8 OFF on RTX 4090")
print("="*80)
print()

# Test configuration
test_video = 'videostotrain/first.mp4'
output_dir_baseline = 'results/benchmark_baseline'
output_dir_fp8 = 'results/benchmark_fp8'

if not os.path.exists(test_video):
    print(f"[ERROR] Test video not found: {test_video}")
    exit(1)

os.makedirs(output_dir_baseline, exist_ok=True)
os.makedirs(output_dir_fp8, exist_ok=True)

# Load video frames (limit to 30 frames for quick benchmark)
import cv2
import numpy as np

cap = cv2.VideoCapture(test_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames = []
frame_limit = 30
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

# Create dummy masks (20% masked area)
masks = []
for frame in frames:
    mask = np.zeros((height, width), dtype=np.uint8)
    h_start, h_end = height//3, 2*height//3
    w_start, w_end = width//3, 2*width//3
    mask[h_start:h_end, w_start:w_end] = 255
    masks.append(mask)

frames_array = [np.ascontiguousarray(f) for f in frames]
masks_array = [np.ascontiguousarray(m) for m in masks]

print("="*80)
print("TEST 1: BASELINE (FP8 OFF)")
print("="*80)
print()

# Configure environment for baseline
os.environ['USE_NEUFLOW'] = '1'
os.environ['ENABLE_FLASH_ATTENTION'] = '1'
os.environ['ENABLE_FP8_TRANSFORMER'] = '0'  # FP8 OFF

# Import pipeline (must import AFTER setting environment variables)
from watermark import pipeline as propainter_pipeline

print("Running baseline (FP8 OFF)...")
start_baseline = time.time()
try:
    propainter_pipeline(
        video=test_video,
        mask='dummy',
        output=output_dir_baseline,
        resize_ratio=1.0,
        mask_dilation=4,
        ref_stride=15,
        neighbor_length=10,
        subvideo_length=120,
        raft_iter=10,
        mode="video_inpainting",
        save_fps=fps,
        save_frames=True,
        fp16=False,  # Disable FP16 to avoid dtype issues (testing FP8 only)
        use_cached_models=False,  # Force reload to apply FP8 settings
        frames_array=frames_array,
        masks_array=masks_array
    )
except Exception as e:
    print(f"[ERROR] Baseline failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

baseline_time = time.time() - start_baseline
print(f"\n[BASELINE] Total time: {baseline_time:.2f}s")
print(f"[BASELINE] Per-frame: {(baseline_time/len(frames))*1000:.1f}ms")
print()

print("="*80)
print("TEST 2: FP8 QUANTIZATION (FP8 ON)")
print("="*80)
print()

# Configure environment for FP8
os.environ['ENABLE_FP8_TRANSFORMER'] = '1'  # FP8 ON

# Clear module cache to reload with FP8
import importlib
if 'model.modules.sparse_transformer' in sys.modules:
    del sys.modules['model.modules.sparse_transformer']
if 'model.propainter' in sys.modules:
    del sys.modules['model.propainter']
if 'watermark' in sys.modules:
    del sys.modules['watermark']

# Reimport pipeline with FP8 enabled
from watermark import pipeline as propainter_pipeline_fp8

print("Running with FP8 quantization (FP8 ON)...")
start_fp8 = time.time()
try:
    propainter_pipeline_fp8(
        video=test_video,
        mask='dummy',
        output=output_dir_fp8,
        resize_ratio=1.0,
        mask_dilation=4,
        ref_stride=15,
        neighbor_length=10,
        subvideo_length=120,
        raft_iter=10,
        mode="video_inpainting",
        save_fps=fps,
        save_frames=True,
        fp16=False,  # Disable FP16 to avoid dtype issues (testing FP8 only)
        use_cached_models=False,  # Force reload
        frames_array=frames_array,
        masks_array=masks_array
    )
except Exception as e:
    print(f"[ERROR] FP8 test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

fp8_time = time.time() - start_fp8
print(f"\n[FP8] Total time: {fp8_time:.2f}s")
print(f"[FP8] Per-frame: {(fp8_time/len(frames))*1000:.1f}ms")
print()

# Calculate speedup
speedup = baseline_time / fp8_time if fp8_time > 0 else 0
time_saved = baseline_time - fp8_time

print("="*80)
print("RESULTS COMPARISON")
print("="*80)
print()
print(f"{'Metric':<30} {'Baseline (FP8 OFF)':<20} {'FP8 Quantization':<20} {'Speedup':<15}")
print("-"*85)
print(f"{'Total Time':<30} {baseline_time:>18.2f}s {fp8_time:>18.2f}s {speedup:>13.2f}x")
print(f"{'Per-Frame Time':<30} {(baseline_time/len(frames))*1000:>17.1f}ms {(fp8_time/len(frames))*1000:>17.1f}ms")
print(f"{'Time Saved':<30} {'':<20} {time_saved:>18.2f}s")
print(f"{'Speed Improvement':<30} {'':<20} {((speedup-1)*100):>17.1f}%")
print()

if speedup >= 1.3:
    print(f"✓ FP8 SUCCESSFUL! {speedup:.2f}x speedup achieved (expected: 1.3-1.5x)")
elif speedup >= 1.1:
    print(f"⚠ FP8 working but lower than expected: {speedup:.2f}x (expected: 1.3-1.5x)")
    print("  This might be due to overhead from other components")
else:
    print(f"⚠ FP8 speedup minimal: {speedup:.2f}x")
    print("  Check if FP8 was actually enabled (look for FP8 messages above)")

print()
print("="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print()
print(f"Baseline output: {output_dir_baseline}")
print(f"FP8 output: {output_dir_fp8}")
print()
print("Compare the output videos to verify quality is maintained!")
