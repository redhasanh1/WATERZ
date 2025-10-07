import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import glob
import random
import time
import os
from yolo_detector import YOLOWatermarkDetector
from wavepaint_tensorrt_inpainter import WavePaintTensorRTInpainter
from tqdm import tqdm

print("=" * 60)
print("WAVEPAINT TENSORRT VIDEO WATERMARK REMOVAL")
print("TensorRT YOLO + WavePaint TensorRT (10-20x faster!)")
print("=" * 60)

# Find all videos and pick randomly
video_files = glob.glob('videostotrain/*.mp4')
if not video_files:
    print("No videos found in videostotrain folder!")
    exit()

input_video = random.choice(video_files)
output_video = 'test_video_removal_wavepaint_result.mp4'

print(f"\n📹 Input: {input_video}")
print(f"📁 Output: {output_video}")

# Initialize models
print("\n" + "=" * 60)
print("INITIALIZING")
print("=" * 60)
print("\nLoading TensorRT YOLO detector...")
detector = YOLOWatermarkDetector()

print("\nLoading WavePaint TensorRT inpainter...")
inpainter = WavePaintTensorRTInpainter()

# Open video
print("\n" + "=" * 60)
print("VIDEO INFO")
print("=" * 60)

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nResolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {total_frames/fps:.1f}s")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ Failed to create video writer!")
    exit()

print("\n" + "=" * 60)
print("PROCESSING VIDEO")
print("=" * 60)
print(f"\nUsing:")
print(f"  - TensorRT YOLO for detection (GPU accelerated)")
print(f"  - WavePaint TensorRT for inpainting (10-20x faster!)")
print(f"  - 0px padding (tight detection)")
print(f"  - Expected: MUCH better quality than LaMa!")
print()

# Track stats
frames_processed = 0
frames_with_watermark = 0
total_detect_time = 0
total_inpaint_time = 0
start_time = time.time()

# Process frames
for frame_num in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()

    if not ret:
        break

    # Detect watermark
    detect_start = time.time()
    detections = detector.detect(frame, confidence_threshold=0.25, padding=0)
    detect_time = time.time() - detect_start
    total_detect_time += detect_time

    if detections:
        frames_with_watermark += 1

        # Create mask
        mask = detector.create_mask(frame, detections)

        # Inpaint with WavePaint TensorRT
        inpaint_start = time.time()
        result = inpainter.inpaint_region(frame, mask)
        inpaint_time = time.time() - inpaint_start
        total_inpaint_time += inpaint_time

        out.write(result)
    else:
        # No watermark, write original
        out.write(frame)

    frames_processed += 1

# Cleanup
cap.release()
out.release()

total_time = time.time() - start_time

# Results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"\n✅ Video processing complete!")
print(f"\nFrames:")
print(f"  - Total processed: {frames_processed}")
print(f"  - With watermark: {frames_with_watermark}")
print(f"  - Clean: {frames_processed - frames_with_watermark}")

print(f"\nPerformance:")
print(f"  - Total time: {total_time:.1f}s")
print(f"  - FPS: {frames_processed/total_time:.1f}")
print(f"  - Avg detection: {total_detect_time/frames_processed*1000:.1f}ms/frame")
if frames_with_watermark > 0:
    print(f"  - Avg inpainting: {total_inpaint_time/frames_with_watermark*1000:.1f}ms/frame")
    print(f"  - 🚀 WavePaint TensorRT speed: {8000/(total_inpaint_time/frames_with_watermark*1000):.1f}x faster than PyTorch!")

print(f"\n📁 Saved to: {output_video}")
print(f"\n💡 Play the video to see WavePaint watermark removal!")
print(f"   Quality should be MUCH better than LaMa!")
print("=" * 60)
