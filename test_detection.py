import sys
sys.path.insert(0, 'python_packages')

import cv2
from yolo_detector import YOLOWatermarkDetector

# Find all videos in videostotrain folder
import glob
import random
video_files = glob.glob('videostotrain/*.mp4')
if not video_files:
    print("No videos found in videostotrain folder!")
    exit()

# Pick random video
video_path = random.choice(video_files)
print(f"Testing on: {video_path}\n")

# Pick random frame from the video
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
random_frame_num = random.randint(0, max(0, total_frames - 1))
cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_num)
print(f"Using random frame {random_frame_num}/{total_frames}\n")

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
print("Testing with confidence threshold 0.25 and 5px padding...")
detections = detector.detect(frame, confidence_threshold=0.25, padding=5)

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
