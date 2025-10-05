import cv2
import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, 'python_packages')

# Read template
template = cv2.imread('watermark_template.png')
if template is None:
    print("Error: watermark_template.png not found!")
    print("Run crop_template.py first!")
    exit()

th, tw = template.shape[:2]
print(f"Using template: {tw}x{th}")

# Open video
video_path = 'sora_with_watermark.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video: {video_path}")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

# Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sora_no_watermark.mp4', fourcc, fps, (width, height))

frame_num = 0
success_count = 0

print("\nProcessing video...")
with tqdm(total=total_frames) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Find watermark
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.6:  # Good match
            top_left = max_loc
            bottom_right = (top_left[0] + tw, top_left[1] + th)

            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255

            # Remove watermark
            frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            success_count += 1

        out.write(frame)
        frame_num += 1
        pbar.update(1)

cap.release()
out.release()

print(f"\n✅ Done! Processed {total_frames} frames")
print(f"Watermark removed from {success_count} frames ({success_count/total_frames*100:.1f}%)")
print(f"Output: sora_no_watermark.mp4")
