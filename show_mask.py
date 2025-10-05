import cv2
import numpy as np

# Read the video frame
cap = cv2.VideoCapture('sora_with_watermark.mp4')
ret, frame = cap.read()
cap.release()

if ret:
    h, w = frame.shape[:2]

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Sora watermark in top-middle
    watermark_width = int(w * 0.3)
    watermark_height = int(h * 0.08)
    watermark_x = int(w * 0.35)
    watermark_y = 0
    mask[watermark_y:watermark_height, watermark_x:watermark_x+watermark_width] = 255

    # Overlay mask on frame to visualize
    overlay = frame.copy()
    overlay[mask == 255] = [0, 255, 0]  # Green where mask is
    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    cv2.imwrite('mask_preview.jpg', result)
    print(f"Saved mask_preview.jpg - green area shows what will be removed")
    print(f"Mask area: x={watermark_x}, y={watermark_y} to bottom-right corner")
    print(f"Video size: {w}x{h}")
    print("\nLook at mask_preview.jpg - the GREEN area is what gets removed!")
