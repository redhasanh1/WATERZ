import cv2
import sys

frame_number = int(sys.argv[1]) if len(sys.argv) > 1 else 100

cap = cv2.VideoCapture('sora_with_watermark.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imwrite('test_frame.jpg', frame)
    print(f"Saved frame {frame_number} to test_frame.jpg")
else:
    print(f"Error reading frame {frame_number}")
