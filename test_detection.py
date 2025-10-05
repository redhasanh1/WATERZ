import sys
sys.path.insert(0, 'python_packages')

import cv2
from yolo_detector import YOLOWatermarkDetector

# Find latest video in videostotrain folder
import glob
video_files = glob.glob('videostotrain/*.mp4')
if not video_files:
    print("No videos found in videostotrain folder!")
    exit()

video_path = sorted(video_files)[-1]  # Use newest
print(f"Testing on: {video_path}\n")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video!")
    exit()

print("Initializing YOLO detector...")
detector = YOLOWatermarkDetector()

print(f"\nModel type: {type(detector.model)}")
print(f"Model info: {detector.model}")

print("\nDetecting watermarks with NEW trained model (.pt, not TensorRT yet)...")
print("Testing with confidence threshold 0.25...")
detections = detector.detect(frame, confidence_threshold=0.25, padding=30)

if not detections:
    print("\n❌ No watermarks detected!")
else:
    print(f"\n✅ Found {len(detections)} detection(s)")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        print(f"  {i+1}. Box: ({x1},{y1}) to ({x2},{y2}), Confidence: {conf:.2%}")

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{conf:.2%}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite('test_detection_result.jpg', frame)
print("\nSaved test_detection_result.jpg - check what it detected!")
