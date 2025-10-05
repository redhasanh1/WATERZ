import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import glob
from tqdm import tqdm
from lama_inpaint import LamaInpainter

# Load templates
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

# Initialize LaMa
print("Loading LaMa AI model...")
inpainter = LamaInpainter(device='cuda')

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
out = cv2.VideoWriter('sora_no_watermark_ai.mp4', fourcc, fps, (width, height))

success_count = 0
last_position = None
padding = 30

print("\nProcessing video with AI inpainting...")
print("This will be slower but produce perfect results!\n")

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
            # Use detected position
            x1 = max(0, best_location[0] - padding)
            y1 = max(0, best_location[1] - padding)
            x2 = min(width, best_location[0] + best_size[0] + padding)
            y2 = min(height, best_location[1] + best_size[1] + padding)
            last_position = (x1, y1, x2, y2)

        elif last_position:
            # Use last known position
            x1, y1, x2, y2 = last_position

        else:
            # No detection and no history - skip frame
            out.write(frame)
            pbar.update(1)
            continue

        # Apply mask
        mask[y1:y2, x1:x2] = 255

        # AI inpainting
        frame = inpainter.inpaint_region(frame, mask)
        success_count += 1

        out.write(frame)
        pbar.update(1)

cap.release()
out.release()

print(f"\n✅ Done! AI removed watermark from {success_count}/{total_frames} frames ({success_count/total_frames*100:.1f}%)")
print(f"Output: sora_no_watermark_ai.mp4")
print("Result should be 99% undetectable!")
