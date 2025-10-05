import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import glob
from tqdm import tqdm
from sd_inpaint import SDInpainter

# Load templates
template_files = glob.glob('templates/template_*.png')
if not template_files:
    print("No templates! Run create_templates.py first")
    exit()

templates = []
for tf in template_files:
    t = cv2.imread(tf)
    if t is not None:
        templates.append(t)

print(f"Loaded {len(templates)} templates")

# Initialize SD inpainting
print("\nLoading Stable Diffusion model...")
inpainter = SDInpainter(device='cuda')

# Open video
video_path = 'asser.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: asser.mp4 not found!")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {width}x{height} @ {fps}fps, {total_frames} frames")
print(f"Estimated time: {total_frames * 30 / 60:.1f} minutes\n")

# Create output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('asser_clean.mp4', fourcc, fps, (width, height))

success_count = 0
last_position = None
padding = 30

print("Processing asser.mp4 with SD AI...")
with tqdm(total=total_frames) as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect watermark
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

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)

        if best_confidence > 0.5:
            x1 = max(0, best_location[0] - padding)
            y1 = max(0, best_location[1] - padding)
            x2 = min(width, best_location[0] + best_size[0] + padding)
            y2 = min(height, best_location[1] + best_size[1] + padding)
            last_position = (x1, y1, x2, y2)

        elif last_position:
            x1, y1, x2, y2 = last_position

        else:
            out.write(frame)
            pbar.update(1)
            continue

        mask[y1:y2, x1:x2] = 255

        # SD inpainting
        frame = inpainter.inpaint_region(frame, mask)
        success_count += 1

        out.write(frame)
        pbar.update(1)

cap.release()
out.release()

print(f"\n✅ Done! Removed watermark from {success_count}/{total_frames} frames ({success_count/total_frames*100:.1f}%)")
print(f"Output: asser_clean.mp4")
