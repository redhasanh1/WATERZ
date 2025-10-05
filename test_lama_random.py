import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import glob
import random
from lama_inpaint_local import LamaInpainter

# Load templates
template_files = glob.glob('templates/*.png') + glob.glob('watermark_template.png')
templates = []
for tf in template_files:
    t = cv2.imread(tf)
    if t is not None:
        templates.append(t)

print(f"Using {len(templates)} templates")

# Pick random frame
video_path = 'asser.mp4'
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

random_frame_num = random.randint(50, total_frames - 50)
print(f"\nTesting on random frame {random_frame_num}/{total_frames}")

cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_num)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading frame!")
    exit()

# Save test frame
cv2.imwrite('test_frame_lama.jpg', frame)

h, w = frame.shape[:2]

# Detect watermark
print("Detecting watermark...")
best_confidence = 0
best_location = None
best_size = None

for template in templates:
    th, tw = template.shape[:2]
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > best_confidence:
        best_confidence = max_val
        best_location = max_loc
        best_size = (tw, th)

print(f"Detection confidence: {best_confidence:.2%}")

if best_confidence < 0.5:
    print("❌ No watermark detected on this frame!")
    exit()

# Add padding
padding = 30
x1 = max(0, best_location[0] - padding)
y1 = max(0, best_location[1] - padding)
x2 = min(w, best_location[0] + best_size[0] + padding)
y2 = min(h, best_location[1] + best_size[1] + padding)

# Create mask
mask = np.zeros((h, w), dtype=np.uint8)
mask[y1:y2, x1:x2] = 255

# Show detection
detection = frame.copy()
cv2.rectangle(detection, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imwrite('detection_lama.jpg', detection)
print(f"Saved detection_lama.jpg - green box shows what will be removed")

# LaMa inpainting
print("\nRemoving watermark with LaMa AI...")
inpainter = LamaInpainter()
result = inpainter.inpaint_region(frame, mask)

# Save
cv2.imwrite('result_lama_test.jpg', result)
print("✅ Saved result_lama_test.jpg")

# Comparison
comparison = np.hstack([frame, result])
cv2.imwrite('comparison_lama_test.jpg', comparison)
print("Saved comparison_lama_test.jpg (before | after)")
print(f"\nFrame {random_frame_num} processed with LaMa!")
