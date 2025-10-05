import cv2
import numpy as np

# Read the test frame
frame = cv2.imread('test_frame.jpg')

if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply threshold to find bright text/logo
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw green boxes around detected text areas
result = frame.copy()
detected_areas = []

for cnt in contours:
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    # Filter small noise and very large areas
    if w_box > 20 and h_box > 10 and w_box < w*0.5 and h_box < h*0.3:
        cv2.rectangle(result, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
        detected_areas.append((x, y, w_box, h_box))
        print(f"Found potential watermark at: x={x}, y={y}, width={w_box}, height={h_box}")

cv2.imwrite('watermark_detection.jpg', result)
print(f"\nSaved watermark_detection.jpg - green boxes show detected areas")
print(f"Total areas found: {len(detected_areas)}")
