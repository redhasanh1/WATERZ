import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_local import LamaInpainter
import os

print("=" * 60)
print("YOLOv8 + LaMa Custom Frame Test")
print("=" * 60)

# Ask for video
print("\nWhich video?")
print("1. sora_with_watermark.mp4")
print("2. asser.mp4")
video_choice = input("Enter choice (1/2): ").strip()

if video_choice == '1':
    video_path = 'sora_with_watermark.mp4'
else:
    video_path = 'asser.mp4'

if not os.path.exists(video_path):
    print(f"❌ {video_path} not found!")
    exit()

# Open video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {video_path}")
print(f"Total frames: {total_frames}")

# Random frame
import random
frame_num = random.randint(0, total_frames - 1)
print(f"\nTesting on random frame {frame_num}/{total_frames}")

# Extract frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Error reading frame!")
    exit()

print(f"\n✅ Loaded frame {frame_num}")

# Save original
cv2.imwrite('test_frame_custom.jpg', frame)
print("Saved test_frame_custom.jpg")

# Detect watermark
print("\n" + "=" * 60)
print("STEP 1: YOLOv8 Watermark Detection")
print("=" * 60)

detector = YOLOWatermarkDetector()
detections = detector.detect(frame, confidence_threshold=0.3, padding=30)

if not detections:
    print("\n❌ No watermarks detected!")
    print("Try a different frame or lower confidence threshold")
else:
    print(f"\n✅ Found {len(detections)} watermark(s)")

    # Draw detection
    detection_vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite('detection_custom.jpg', detection_vis)
    print("Saved detection_custom.jpg")

    # Create mask
    mask = detector.create_mask(frame, detections)
    cv2.imwrite('mask_custom.png', mask)

    # LaMa inpainting
    print("\n" + "=" * 60)
    print("STEP 2: LaMa Inpainting")
    print("=" * 60)

    inpainter = LamaInpainter()
    result = inpainter.inpaint_region(frame, mask)

    if result is not None:
        cv2.imwrite('result_custom.jpg', result)
        print("✅ Saved result_custom.jpg")

        # Create comparison
        h, w = frame.shape[:2]
        comparison = np.zeros((h, w*3, 3), dtype=np.uint8)
        comparison[:, :w] = frame
        comparison[:, w:w*2] = detection_vis
        comparison[:, w*2:] = result

        # Add labels
        cv2.putText(comparison, "Original", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(comparison, "Detection", (w+10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(comparison, "Removed", (w*2+10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        cv2.imwrite('comparison_custom.jpg', comparison)

        print("\n" + "=" * 60)
        print("✅ SUCCESS!")
        print("=" * 60)
        print("Saved comparison_custom.jpg")
    else:
        print("❌ LaMa inpainting failed!")

print("\nDone!")
