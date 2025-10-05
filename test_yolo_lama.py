import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import random
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_local import LamaInpainter

# Pick random frame
video_path = 'asser.mp4'
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

random_frame_num = random.randint(50, total_frames - 50)
print(f"Testing on random frame {random_frame_num}/{total_frames}\n")

cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_num)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading frame!")
    exit()

# Save test frame
cv2.imwrite('test_frame_yolo.jpg', frame)
print("Saved test_frame_yolo.jpg\n")

# Initialize YOLOv8 detector
print("=" * 50)
print("STEP 1: YOLOv8 Watermark Detection")
print("=" * 50)
detector = YOLOWatermarkDetector()

# Detect watermarks
detections = detector.detect(frame, confidence_threshold=0.25, padding=30)

if not detections:
    print("\n❌ No watermarks detected by YOLOv8!")
    print("The model might need training on your specific watermark type")
    exit()

print(f"\n✅ Found {len(detections)} watermark(s)")

# Create mask from detections
mask = detector.create_mask(frame, detections)

# Show detection
detection_vis = frame.copy()
for det in detections:
    x1, y1, x2, y2 = det['bbox']
    conf = det['confidence']
    cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(detection_vis, f"{conf:.2%}", (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite('detection_yolo_lama.jpg', detection_vis)
print("Saved detection_yolo_lama.jpg - green boxes show detected watermarks")

# LaMa inpainting
print("\n" + "=" * 50)
print("STEP 2: LaMa AI Watermark Removal")
print("=" * 50)
inpainter = LamaInpainter()
result = inpainter.inpaint_region(frame, mask)

# Save result
cv2.imwrite('result_yolo_lama.jpg', result)
print("\n✅ Saved result_yolo_lama.jpg")

# Comparison
comparison = np.hstack([frame, result])
cv2.imwrite('comparison_yolo_lama.jpg', comparison)
print("Saved comparison_yolo_lama.jpg (before | after)")

print("\n" + "=" * 50)
print(f"Frame {random_frame_num} processed with YOLOv8 + LaMa!")
print("=" * 50)
print("\nCheck the results:")
print("  - detection_yolo_lama.jpg: What was detected")
print("  - result_yolo_lama.jpg: Final cleaned result")
print("  - comparison_yolo_lama.jpg: Side-by-side comparison")
