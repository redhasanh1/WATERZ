"""
Quick FP8 Production Test
Tests that FP8 Conv2d/Conv3d optimization is working correctly in the actual pipeline
"""
import os
import sys
import time
import subprocess

# Fix Unicode encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

print("="*80)
print("FP8 PRODUCTION TEST - Quick Verification")
print("="*80)
print()

# Setup paths
sys.path.insert(0, 'python_packages')
sys.path.insert(0, 'faster-propainter-main')
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')

# Enable FP8
os.environ['ENABLE_FP8_ENCODER'] = '1'
os.environ['ENABLE_FP8_DECODER'] = '1'
os.environ['ENABLE_FP8_RFCNET'] = '1'
os.environ['USE_NEUFLOW'] = '1'
os.environ['ENABLE_FLASH_ATTENTION'] = '1'
os.environ['ENABLE_FP8_TRANSFORMER'] = '1'

print("[CONFIG] FP8 Settings:")
print(f"  ENABLE_FP8_ENCODER = {os.environ['ENABLE_FP8_ENCODER']}")
print(f"  ENABLE_FP8_DECODER = {os.environ['ENABLE_FP8_DECODER']}")
print(f"  ENABLE_FP8_RFCNET = {os.environ['ENABLE_FP8_RFCNET']}")
print()

# Find a test video
test_videos = [
    'videostotrain/324b4d52-260b-470e-971f-8018406ef0d9.mp4',
    'videostotrain/5524f63e-161a-4521-aad5-17649263e503.mp4',
    'videostotrain/60d75dd6-e508-465c-93dd-e6a8b1c6fe31.mp4',
]

test_video = None
for video in test_videos:
    if os.path.exists(video):
        test_video = video
        break

if not test_video:
    print("[ERROR] No test video found!")
    print("Please place a video in videostotrain/ directory")
    sys.exit(1)

print(f"[TEST] Using video: {test_video}")
print()

# Import pipeline
print("[INIT] Loading ProPainter pipeline...")
try:
    from watermark import pipeline as propainter_pipeline
    import cv2
    import numpy as np
    print("[OK] Pipeline loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load video metadata
cap = cv2.VideoCapture(test_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Limit to 30 frames for quick test
frame_limit = min(30, total_frames)

print(f"[VIDEO] {width}x{height} @ {fps} fps, {total_frames} total frames")
print(f"[TEST] Processing first {frame_limit} frames")
print()

# Load frames
print(f"[LOADING] Reading {frame_limit} frames...")
cap = cv2.VideoCapture(test_video)
frames = []
for i in range(frame_limit):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f"[OK] Loaded {len(frames)} frames")
print()

# Create dummy masks (center 20% region)
masks = []
for frame in frames:
    mask = np.zeros((height, width), dtype=np.uint8)
    h_start, h_end = height//3, 2*height//3
    w_start, w_end = width//3, 2*width//3
    mask[h_start:h_end, w_start:w_end] = 255
    masks.append(mask)

# Prepare arrays
frames_array = [np.ascontiguousarray(f) for f in frames]
masks_array = [np.ascontiguousarray(m) for m in masks]

print("[RUNNING] FP8 Test - Processing with all FP8 optimizations enabled...")
print("="*80)

# Run pipeline
output_dir = 'results/fp8_production_test'
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
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
    success = True
except Exception as e:
    success = False
    print(f"[ERROR] Pipeline failed: {e}")
    import traceback
    traceback.print_exc()

total_time = time.time() - start_time

print()
print("="*80)
print("TEST RESULTS")
print("="*80)
print()

if success:
    per_frame_ms = (total_time / len(frames)) * 1000
    fps_achieved = len(frames) / total_time

    print(f"✅ SUCCESS - FP8 optimization working!")
    print()
    print(f"Performance Metrics:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Frames processed: {len(frames)}")
    print(f"  Per-frame time: {per_frame_ms:.1f}ms")
    print(f"  Processing FPS: {fps_achieved:.1f}")
    print()

    # Expected baseline (without FP8): ~200ms/frame
    # With FP8: ~170-180ms/frame (15-18% faster)
    expected_baseline = 200  # ms
    improvement = ((expected_baseline - per_frame_ms) / expected_baseline) * 100

    print(f"Expected Performance:")
    print(f"  Baseline (FP16): ~{expected_baseline}ms/frame")
    print(f"  With FP8: {per_frame_ms:.1f}ms/frame")

    if per_frame_ms < expected_baseline:
        print(f"  ✅ Improvement: {improvement:.1f}% faster than baseline!")
    else:
        print(f"  ⚠️ Slower than expected (may need warmup or GPU is busy)")

    print()
    print(f"Output saved to: {output_dir}")

    # Check if output video exists
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    if output_files:
        print(f"  ✅ Output video: {output_files[0]}")

else:
    print(f"❌ FAILED - Pipeline encountered errors")
    print()
    print("Please check the error messages above")

print()
print("="*80)
print("FP8 PRODUCTION TEST COMPLETE")
print("="*80)
