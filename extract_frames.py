import sys
import os
import cv2

if len(sys.argv) < 2:
    print("Usage: python extract_frames.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
output_dir = 'NEW_SORA_TRAINING/images'

os.makedirs(output_dir, exist_ok=True)

print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video!")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print(f"\nExtracting every 5th frame (~6 frames per second for better training)...")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save every 30th frame (1 per second)
    if frame_count % 30 == 0:
        output_path = os.path.join(output_dir, f'sora_frame_{saved_count:04d}.jpg')
        cv2.imwrite(output_path, frame)
        saved_count += 1
        print(f"Saved frame {frame_count}/{total_frames} -> {output_path}")

    frame_count += 1

cap.release()

print(f"\n✅ Extracted {saved_count} frames to {output_dir}/")
print("\nNext step: Label the watermarks using labelImg or Roboflow!")
print("Or use our auto-labeling tool if the watermark is consistent.")
