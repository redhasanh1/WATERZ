import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import glob
from tqdm import tqdm

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

print(f"Using {len(templates)} templates")

# Open video
cap = cv2.VideoCapture('sora_with_watermark.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

# Create output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sora_clean_final.mp4', fourcc, fps, (width, height))

padding = 40  # Larger padding for better context
success_count = 0
last_position = None

print("\nProcessing with improved removal...")

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

        if best_confidence > 0.45:  # Lower threshold
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

        # Multi-pass inpainting for better quality
        # Pass 1: NS method
        frame = cv2.inpaint(frame, mask, 20, cv2.INPAINT_NS)

        # Pass 2: TELEA method for smoothing
        frame = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)

        success_count += 1
        out.write(frame)
        pbar.update(1)

cap.release()
out.release()

print(f"\n✅ Processed {success_count}/{total_frames} frames ({success_count/total_frames*100:.1f}%)")
print("Output: sora_clean_final.mp4")
print("\nImprovements:")
print("- Lower detection threshold (45% vs 50%)")
print("- Larger padding (40px vs 30px)")
print("- Multi-pass inpainting (NS + TELEA)")
print("- Better temporal tracking")
