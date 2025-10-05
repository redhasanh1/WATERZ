import cv2
import numpy as np

# Global variables
drawing = False
ix, iy = -1, -1
mask = None
img_copy = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, mask, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_display = img_copy.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Draw Watermark Area - Press SPACE when done, C to clear', img_display)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.rectangle(mask, (ix, iy), (x, y), 255, -1)
        cv2.imshow('Draw Watermark Area - Press SPACE when done, C to clear', img_copy)

# Read first frame
cap = cv2.VideoCapture('sora_with_watermark.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video!")
    exit()

h, w = frame.shape[:2]
mask = np.zeros((h, w), dtype=np.uint8)
img_copy = frame.copy()

cv2.namedWindow('Draw Watermark Area - Press SPACE when done, C to clear')
cv2.setMouseCallback('Draw Watermark Area - Press SPACE when done, C to clear', draw_rectangle)
cv2.imshow('Draw Watermark Area - Press SPACE when done, C to clear', frame)

print("Instructions:")
print("1. Click and drag to draw rectangle(s) over the watermark")
print("2. Press 'c' to clear all rectangles")
print("3. Press SPACE when done")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Space bar
        break
    elif key == ord('c'):  # Clear
        mask = np.zeros((h, w), dtype=np.uint8)
        img_copy = frame.copy()
        cv2.imshow('Draw Watermark Area - Press SPACE when done, C to clear', img_copy)

cv2.destroyAllWindows()

# Save mask
cv2.imwrite('mask.png', mask)
print("Mask saved to mask.png")

# Show preview
overlay = frame.copy()
overlay[mask == 255] = [0, 255, 0]
result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
cv2.imwrite('mask_preview.jpg', result)
print("Preview saved to mask_preview.jpg (green = will be removed)")
