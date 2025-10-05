import cv2
import numpy as np
import sys

# Read template
template = cv2.imread('watermark_template.png')
if template is None:
    print("Error: watermark_template.png not found!")
    print("Run crop_template.py first to create the template")
    exit()

# Read frame to search
frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]
th, tw = template.shape[:2]

print(f"Searching for {tw}x{th} template in {w}x{h} frame...")

# Template matching
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Best match location
top_left = max_loc
bottom_right = (top_left[0] + tw, top_left[1] + th)

print(f"Match confidence: {max_val:.2%}")
print(f"Watermark found at: x={top_left[0]}, y={top_left[1]}")

# Draw green box
detection = frame.copy()
cv2.rectangle(detection, top_left, bottom_right, (0, 255, 0), 3)
cv2.imwrite('watermark_detection.jpg', detection)

# Create mask
mask = np.zeros((h, w), dtype=np.uint8)
mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
cv2.imwrite('mask.png', mask)

print("\nSaved watermark_detection.jpg (green box shows detection)")
print("Saved mask.png (ready for removal)")
print(f"\nConfidence: {max_val:.2%} - {'GOOD' if max_val > 0.7 else 'LOW (may need better template)'}")
