import cv2
import numpy as np
import glob

# Load all templates
template_files = glob.glob('templates/template_*.png')

if not template_files:
    print("No templates found! Run create_templates.py first")
    exit()

templates = []
for tf in template_files:
    t = cv2.imread(tf)
    if t is not None:
        templates.append(t)

print(f"Using {len(templates)} templates")

# Read frame
frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]

# Try all templates
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

print(f"Confidence: {best_confidence:.2%}")

if best_confidence > 0.5:
    # Add padding (30 pixels on all sides)
    padding = 30
    x1 = max(0, best_location[0] - padding)
    y1 = max(0, best_location[1] - padding)
    x2 = min(w, best_location[0] + best_size[0] + padding)
    y2 = min(h, best_location[1] + best_size[1] + padding)

    print(f"Watermark at: ({best_location[0]}, {best_location[1]})")
    print(f"With padding: ({x1}, {y1}) to ({x2}, {y2})")

    # Draw detection with padding
    detection = frame.copy()
    # Original detection (red)
    cv2.rectangle(detection, best_location,
                  (best_location[0]+best_size[0], best_location[1]+best_size[1]),
                  (0, 0, 255), 2)
    # Padded region (green)
    cv2.rectangle(detection, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('watermark_detection.jpg', detection)

    # Create mask with padding
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    cv2.imwrite('mask.png', mask)

    print("\n✓ Detection saved!")
    print("Red box = exact watermark")
    print("Green box = padded region for AI inpainting")
else:
    print("\n✗ No watermark detected")
