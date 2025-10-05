import cv2
import numpy as np

frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Try multiple threshold levels to detect dark or light text
result = frame.copy()
all_boxes = []

# Detect dark text
_, thresh_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
contours_dark, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Detect light text
_, thresh_light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
contours_light, _ = cv2.findContours(thresh_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Combine all contours
all_contours = list(contours_dark) + list(contours_light)

for cnt in all_contours:
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    # Look for text-like shapes
    aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
    if (w_box > 15 and h_box > 8 and w_box < w*0.6 and h_box < h*0.2 and
        aspect_ratio > 0.5 and aspect_ratio < 15):
        cv2.rectangle(result, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
        all_boxes.append((x, y, w_box, h_box))
        print(f"Found text at: x={x}, y={y}, width={w_box}, height={h_box}")

cv2.imwrite('watermark_detection.jpg', result)
print(f"\nSaved watermark_detection.jpg")
print(f"Total boxes: {len(all_boxes)}")
print("Green boxes show all detected text - find the Sora one")
