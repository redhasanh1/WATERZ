import cv2
import numpy as np

# Extract first frame
cap = cv2.VideoCapture('sora_with_watermark.mp4')
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite('test_frame.jpg', frame)
    print("Saved test_frame.jpg")

    # Create mask for watermark
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Small watermark area in bottom-right
    watermark_x = int(w * 0.88)
    watermark_y = int(h * 0.94)
    mask[watermark_y:h, watermark_x:w] = 255

    cv2.imwrite('test_mask.png', mask)
    print(f"Saved test_mask.png - watermark area at ({watermark_x}, {watermark_y})")
    print("\nNow install IOPaint and test:")
    print("pip install iopaint")
    print("iopaint start --model=lama --device=cuda --port=8080")
    print("\nThen open http://localhost:8080 and load test_frame.jpg")
