import sys
sys.path.insert(0, 'python_packages')

import cv2
from yolo_detector import YOLOWatermarkDetector

# Test on first uploaded video
video_path = 'uploads/89c6fdbb-742b-4c6d-ad57-e85f37fe13d1.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video!")
    exit()

print("Initializing YOLO detector...")
detector = YOLOWatermarkDetector()

print("\nDetecting watermarks...")
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
