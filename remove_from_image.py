import cv2
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, 'python_packages')

# Read image and mask
frame = cv2.imread('test_frame.jpg')
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

if mask is None:
    print("Error: mask.png not found!")
    exit()

# Resize mask to match frame
if mask.shape != frame.shape[:2]:
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

# Use OpenCV inpainting
result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite('result.jpg', result)
print("Watermark removed! Saved to result.jpg")
