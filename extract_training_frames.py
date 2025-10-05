import sys
sys.path.insert(0, 'python_packages')

import cv2
import os

# Create directory for training data
os.makedirs('yolo_training/images', exist_ok=True)

video_path = 'sora_with_watermark.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

# Extract 40 frames evenly distributed
num_frames = 40
frame_interval = total_frames // num_frames

print(f"Extracting {num_frames} frames (every {frame_interval} frames)...\n")

extracted = 0
for i in range(num_frames):
    frame_num = i * frame_interval
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()

    if ret:
        # Save with zero-padded filename
        filename = f'yolo_training/images/frame_{i:04d}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Saved {filename} (frame {frame_num})")
        extracted += 1

cap.release()

print(f"\n✅ Extracted {extracted} frames to yolo_training/images/")
print("\nNext step: Run create_yolo_labels.py to generate annotations")
