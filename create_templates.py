import cv2
import os

# Create templates directory
os.makedirs('templates', exist_ok=True)

# Frames to extract from
frame_numbers = [0, 50, 100, 150, 200]

cap = cv2.VideoCapture('sora_with_watermark.mp4')

print("Multi-Template Creation")
print("=" * 50)
print("For each frame:")
print("1. Click and drag to select the watermark")
print("2. Press ENTER to save")
print("3. Press C to cancel and skip")
print("=" * 50)

for frame_num in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame {frame_num}")
        continue

    # Resize frame to fit screen (50% of original)
    h, w = frame.shape[:2]
    display_frame = cv2.resize(frame, (w//2, h//2))

    print(f"\nFrame {frame_num}: Select watermark...")

    # Use selectROI on resized frame
    roi = cv2.selectROI(f'Frame {frame_num} - Select watermark, ENTER to save, C to skip', display_frame, False)

    # Scale ROI back to original size
    roi = (roi[0]*2, roi[1]*2, roi[2]*2, roi[3]*2)

    if roi[2] > 0 and roi[3] > 0:  # Valid selection
        template = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        template_path = f'templates/template_{frame_num}.png'
        cv2.imwrite(template_path, template)
        print(f"✓ Saved {template_path} ({int(roi[2])}x{int(roi[3])})")
    else:
        print(f"Skipped frame {frame_num}")

    cv2.destroyAllWindows()

cap.release()

# Count templates
templates = [f for f in os.listdir('templates') if f.endswith('.png')]
print(f"\n{'='*50}")
print(f"Created {len(templates)} templates in ./templates/ folder")
print("You can now use detect_multi_template.py!")
