import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
from tqdm import tqdm
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_local import LamaInpainter

# Initialize YOLOv8 detector
print("=" * 60)
print("Loading AI Models...")
print("=" * 60)
detector = YOLOWatermarkDetector()
inpainter = LamaInpainter()

# Open video
video_path = 'asser.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: asser.mp4 not found!")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {width}x{height} @ {fps}fps, {total_frames} frames")
print(f"Estimated time: ~{total_frames * 1.0 / 60:.1f} minutes (1sec per frame)")

# Create output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('asser_clean_yolo.mp4', fourcc, fps, (width, height))

success_count = 0
detection_count = 0

print("\n" + "=" * 60)
print("Processing video with YOLOv8 + LaMa AI...")
print("=" * 60)

with tqdm(total=total_frames, desc="Processing") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect watermarks with YOLOv8
        detections = detector.detect(frame, confidence_threshold=0.25, padding=30)

        if detections:
            detection_count += 1

            # Create mask from detections
            mask = detector.create_mask(frame, detections)

            # LaMa inpainting
            frame = inpainter.inpaint_region(frame, mask)
            success_count += 1

        out.write(frame)
        pbar.update(1)

cap.release()
out.release()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"✅ Processed {total_frames} frames")
print(f"🎯 Detected watermarks in {detection_count} frames ({detection_count/total_frames*100:.1f}%)")
print(f"🔧 Removed watermarks from {success_count} frames ({success_count/total_frames*100:.1f}%)")
print(f"\nOutput: asser_clean_yolo.mp4")
print("=" * 60)
