import cv2
import numpy as np

# Global variables
drawing = False
ix, iy = -1, -1
img_display = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img_display.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Crop Watermark Template - Press SPACE when done', img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Save the cropped area
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)

        if x2 > x1 and y2 > y1:
            template = img_display[y1:y2, x1:x2]
            cv2.imwrite('watermark_template.png', template)
            print(f"Template saved! Size: {x2-x1}x{y2-y1}")

            # Show preview
            preview = img_display.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Crop Watermark Template - Press SPACE when done', preview)

# Read frame
frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

img_display = frame.copy()

cv2.namedWindow('Crop Watermark Template - Press SPACE when done')
cv2.setMouseCallback('Crop Watermark Template - Press SPACE when done', draw_rectangle)
cv2.imshow('Crop Watermark Template - Press SPACE when done', frame)

print("Instructions:")
print("1. Click and drag to select ONLY the Sora watermark (symbol + text)")
print("2. Release mouse to save template")
print("3. Press SPACE to finish")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nTemplate saved as watermark_template.png")
print("You can now use detect_with_template.py to find it in any frame!")
