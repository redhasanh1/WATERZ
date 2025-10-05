import cv2
import numpy as np
import sys

sys.path.insert(0, 'python_packages')

# Read template
template = cv2.imread('watermark_template.png')
if template is None:
    print("Error: watermark_template.png not found!")
    print("Run crop_template.py first!")
    exit()

# Read frame
frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]
th, tw = template.shape[:2]

# Find watermark
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc
bottom_right = (top_left[0] + tw, top_left[1] + th)

print(f"Watermark detected with {max_val:.2%} confidence at ({top_left[0]}, {top_left[1]})")

# Create mask
mask = np.zeros((h, w), dtype=np.uint8)
mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255

# Remove watermark using inpainting
result_img = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite('result.jpg', result_img)
print("Watermark removed! Saved to result.jpg")
