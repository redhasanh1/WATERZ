import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
from yolo_detector import YOLOWatermarkDetector

print("=" * 60)
print("YOLOv8 Detection Test - Sora Watermark")
print("=" * 60)

# Load Sora video and extract a frame
video_path = 'sora_with_watermark.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: {video_path} not found!")
    exit()

# Get frame from middle of video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
middle_frame = total_frames // 2

cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading frame!")
    exit()

print(f"\nExtracted frame {middle_frame} from {video_path}")
cv2.imwrite('sora_test_frame.jpg', frame)
print("Saved sora_test_frame.jpg\n")

# Initialize YOLOv8 detector
print("Loading YOLOv8 watermark detector...\n")
detector = YOLOWatermarkDetector()

# Detect watermarks
print("Detecting Sora watermark...\n")
detections = detector.detect(frame, confidence_threshold=0.2, padding=30)

if not detections:
    print("=" * 60)
    print("❌ NO WATERMARKS DETECTED")
    print("=" * 60)
    print("\nPossible reasons:")
    print("  - YOLOv8 model not trained on Sora watermarks")
    print("  - Confidence threshold too high")
    print("  - Model needs fine-tuning for this watermark type")
    print("\nTry adjusting confidence_threshold or using template matching")
else:
    print("=" * 60)
    print(f"✅ FOUND {len(detections)} WATERMARK(S)")
    print("=" * 60)

    # Draw green boxes
    result = frame.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']

        print(f"\nWatermark {i+1}:")
        print(f"  Location: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Confidence: {conf:.2%}")

        # Draw green box
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add confidence label
        label = f"{conf:.0%}"
        cv2.putText(result, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite('sora_yolo_detection.jpg', result)
    print("\n" + "=" * 60)
    print("✅ Saved sora_yolo_detection.jpg")
    print("=" * 60)
    print("Check the image to see the green boxes around detected watermarks")

print("\nDone!")
