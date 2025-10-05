import cv2
import numpy as np
import sys
import glob
from tqdm import tqdm

sys.path.insert(0, 'python_packages')

# Load all templates
template_files = glob.glob('templates/template_*.png')

if not template_files:
    print("No templates found! Run create_templates.py first")
    exit()

templates = []
for tf in template_files:
    t = cv2.imread(tf)
    if t is not None:
        templates.append(t)

print(f"Loaded {len(templates)} templates")

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
last_position = None  # Track last known watermark position

print("\nProcessing video...")
with tqdm(total=total_frames) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Try all templates, find best match
        best_confidence = 0
        best_location = None
        best_size = None

        for template in templates:
            th, tw = template.shape[:2]
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence:
                best_confidence = max_val
                best_location = max_loc
                best_size = (tw, th)

        # Use detection if confidence is good enough
        if best_confidence > 0.5:
            top_left = best_location
            bottom_right = (top_left[0] + best_size[0], top_left[1] + best_size[1])
            last_position = (top_left, bottom_right)  # Remember for next frame

            # Create mask and remove
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
            frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            success_count += 1

        elif last_position:  # Use last known position as fallback
            top_left, bottom_right = last_position
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
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
