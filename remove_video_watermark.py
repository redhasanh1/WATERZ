import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_local import LamaInpainter
import os
from tqdm import tqdm

print("=" * 60)
print("YOLOv8 + LaMa Video Watermark Removal")
print("=" * 60)

# Ask for video
print("\nWhich video to process?")
print("1. sora_with_watermark.mp4")
print("2. asser.mp4")
print("3. vid3.mov")
video_choice = input("Enter choice (1/2/3): ").strip()

if video_choice == '1':
    input_video = 'sora_with_watermark.mp4'
    output_video = 'sora_no_watermark_yolo.mp4'
elif video_choice == '2':
    input_video = 'asser.mp4'
    output_video = 'asser_no_watermark_yolo.mp4'
else:
    input_video = 'vid3.mov'
    output_video = 'vid3_no_watermark.mov'

if not os.path.exists(input_video):
    print(f"❌ {input_video} not found!")
    exit()

# Initialize
print("\nInitializing...")
detector = YOLOWatermarkDetector()
inpainter = LamaInpainter()

# Open video
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {input_video}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")

# Setup video writer - use mp4v (works everywhere)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ Failed to open video writer!")
    exit()

print(f"\n🚀 Processing video...")
print("This may take a while...\n")

frames_processed = 0
frames_with_watermark = 0
frames_failed = 0
last_valid_bbox = None  # Track last known watermark position

# Load template for fallback
template = cv2.imread('watermark_template.png')
has_template = template is not None

# Process frames with progress bar
for frame_num in tqdm(range(total_frames), desc="Processing"):
    ret, frame = cap.read()

    if not ret:
        break

    # Detect watermark with lower threshold
    detections = detector.detect(frame, confidence_threshold=0.3, padding=30)

    # Fallback: If YOLO missed it but we have template, try template matching
    if not detections and has_template and last_valid_bbox:
        th, tw = template.shape[:2]
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # If template match is confident enough, use it
        if max_val > 0.6:
            x1, y1 = max_loc
            x2, y2 = x1 + tw, y1 + th
            # Add padding
            x1 = max(0, x1 - 30)
            y1 = max(0, y1 - 30)
            x2 = min(width, x2 + 30)
            y2 = min(height, y2 + 30)
            detections = [{'bbox': (x1, y1, x2, y2), 'confidence': max_val}]

    # If still no detection, use last known position as fallback (temporal consistency)
    if not detections and last_valid_bbox:
        detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]

    if detections:
        frames_with_watermark += 1

        # Update last known position
        if detections[0]['confidence'] > 0.3:
            last_valid_bbox = detections[0]['bbox']

        # Create mask
        mask = detector.create_mask(frame, detections)

        # Remove watermark
        try:
            result = inpainter.inpaint_region(frame, mask)
            out.write(result)
        except Exception as e:
            print(f"\n⚠️  Frame {frame_num} inpainting failed: {e}")
            out.write(frame)  # Write original if failed
            frames_failed += 1
    else:
        # No watermark, write original
        out.write(frame)

    frames_processed += 1

# Cleanup
cap.release()
out.release()

print("\n" + "=" * 60)
print("✅ VIDEO PROCESSING COMPLETE!")
print("=" * 60)
print(f"Frames processed: {frames_processed}")
print(f"Watermarks detected: {frames_with_watermark}")
print(f"Failed inpaints: {frames_failed}")
print(f"\n📁 Saved to: {output_video}")
print("\nYou can now play the video to see watermark removal!")
