import cv2
import numpy as np

frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect text with multiple thresholds
_, thresh_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
_, thresh_light = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

contours_dark, _ = cv2.findContours(thresh_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_light, _ = cv2.findContours(thresh_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

all_contours = list(contours_dark) + list(contours_light)
text_boxes = []

for cnt in all_contours:
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
    if (w_box > 15 and h_box > 8 and w_box < w*0.6 and h_box < h*0.2 and
        aspect_ratio > 0.5 and aspect_ratio < 15):
        text_boxes.append((x, y, w_box, h_box))

if text_boxes:
    # Find bounding box that contains ALL text boxes (the whole watermark)
    min_x = min([b[0] for b in text_boxes])
    min_y = min([b[1] for b in text_boxes])
    max_x = max([b[0] + b[2] for b in text_boxes])
    max_y = max([b[1] + b[3] for b in text_boxes])

    # Add padding to include symbol on the left
    padding_x = int((max_x - min_x) * 0.3)  # 30% padding on sides
    padding_y = int((max_y - min_y) * 0.2)  # 20% padding top/bottom

    min_x = max(0, min_x - padding_x)
    min_y = max(0, min_y - padding_y)
    max_x = min(w, max_x + padding_x)
    max_y = min(h, max_y + padding_y)

    # Draw ONE big green box
    result = frame.copy()
    cv2.rectangle(result, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
    cv2.imwrite('watermark_detection.jpg', result)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[min_y:max_y, min_x:max_x] = 255
    cv2.imwrite('mask.png', mask)

    print(f"Watermark detected at: x={min_x}, y={min_y}, width={max_x-min_x}, height={max_y-min_y}")
    print("Saved watermark_detection.jpg (green box) and mask.png")
else:
    print("No watermark detected!")
