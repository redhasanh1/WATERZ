import cv2
import numpy as np
import os
import glob

# Load all templates
template_files = glob.glob('templates/template_*.png')

if not template_files:
    print("No templates found in ./templates/ folder!")
    print("Run create_templates.py first")
    exit()

templates = []
for tf in template_files:
    t = cv2.imread(tf)
    if t is not None:
        templates.append((tf, t))
        print(f"Loaded {tf}")

print(f"\nUsing {len(templates)} templates")

# Read frame
frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]

# Try all templates
best_match = None
best_confidence = 0
best_location = None
best_size = None

for template_name, template in templates:
    th, tw = template.shape[:2]

    # Template matching
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > best_confidence:
        best_confidence = max_val
        best_match = template_name
        best_location = max_loc
        best_size = (tw, th)

print(f"\nBest match: {best_match}")
print(f"Confidence: {best_confidence:.2%}")
print(f"Location: {best_location}")

if best_confidence > 0.5:  # Lowered threshold
    top_left = best_location
    bottom_right = (top_left[0] + best_size[0], top_left[1] + best_size[1])

    # Draw green box
    detection = frame.copy()
    cv2.rectangle(detection, top_left, bottom_right, (0, 255, 0), 3)
    cv2.imwrite('watermark_detection.jpg', detection)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
    cv2.imwrite('mask.png', mask)

    print("\n✓ Watermark detected!")
    print("Saved watermark_detection.jpg and mask.png")
else:
    print("\n✗ No watermark detected (confidence too low)")
