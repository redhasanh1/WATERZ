import sys
sys.path.insert(0, 'python_packages')

import cv2
import time
import glob
import random
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint import LamaInpainter

print("=" * 60)
print("FULL WATERMARK REMOVAL TEST (TensorRT + LAMA)")
print("=" * 60)

# Find random video
video_files = glob.glob('videostotrain/*.mp4')
if not video_files:
    print("No videos found in videostotrain folder!")
    exit()

video_path = random.choice(video_files)
print(f"\n📹 Video: {video_path}")

# Get random frame
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
random_frame_num = random.randint(0, max(0, total_frames - 1))
cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_num)
print(f"🎬 Frame: {random_frame_num}/{total_frames}")

ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame!")
    exit()

print("\n" + "=" * 60)
print("STEP 1: DETECTION (TensorRT)")
print("=" * 60)

detector = YOLOWatermarkDetector()
print(f"\nModel: {detector.model}")

start_detect = time.time()
detections = detector.detect(frame, confidence_threshold=0.25, padding=0)
detect_time = time.time() - start_detect

if not detections:
    print("\n❌ No watermarks detected!")
    exit()

print(f"\n✅ Found {len(detections)} detection(s) in {detect_time:.3f}s")
for i, det in enumerate(detections):
    x1, y1, x2, y2 = det['bbox']
    conf = det['confidence']
    print(f"  {i+1}. Box: ({x1},{y1}) to ({x2},{y2}), Confidence: {conf:.2%}")

# Draw detection
detection_vis = frame.copy()
for det in detections:
    x1, y1, x2, y2 = det['bbox']
    conf = det['confidence']
    cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(detection_vis, f"{conf:.2%}", (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imwrite('test_detection_result.jpg', detection_vis)

print("\n" + "=" * 60)
print("STEP 2: INPAINTING (LAMA)")
print("=" * 60)

print("\nInitializing LAMA inpainter...")
inpainter = LamaInpainter()

# Create mask
mask = detector.create_mask(frame, detections)
cv2.imwrite('test_mask_result.png', mask)

# Inpaint
start_inpaint = time.time()
result = inpainter.inpaint_region(frame, mask)
inpaint_time = time.time() - start_inpaint

print(f"\n✅ Inpainting complete in {inpaint_time:.3f}s")

# Save result
cv2.imwrite('test_removal_result.jpg', result)

# Create comparison
comparison = cv2.hconcat([frame, result])
cv2.imwrite('test_removal_comparison.jpg', comparison)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total time: {detect_time + inpaint_time:.3f}s")
print(f"  - Detection: {detect_time:.3f}s")
print(f"  - Inpainting: {inpaint_time:.3f}s")
print(f"\nSaved:")
print(f"  - test_detection_result.jpg (detection boxes)")
print(f"  - test_mask_result.png (mask)")
print(f"  - test_removal_result.jpg (final result)")
print(f"  - test_removal_comparison.jpg (before/after)")
print("=" * 60)
