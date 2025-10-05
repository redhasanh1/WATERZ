import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import glob
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

print(f"Using {len(templates)} templates")

# Load frame
frame = cv2.imread('test_frame.jpg')
if frame is None:
    print("Error: test_frame.jpg not found!")
    exit()

h, w = frame.shape[:2]

# Detect watermark
print("Detecting watermark...")
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

if best_confidence < 0.5:
    print("No watermark detected!")
    exit()

print(f"Watermark found with {best_confidence:.2%} confidence")

# Add padding
padding = 30
x1 = max(0, best_location[0] - padding)
y1 = max(0, best_location[1] - padding)
x2 = min(w, best_location[0] + best_size[0] + padding)
y2 = min(h, best_location[1] + best_size[1] + padding)

# Create mask
mask = np.zeros((h, w), dtype=np.uint8)
mask[y1:y2, x1:x2] = 255

# Initialize SD and remove watermark
print("\nRemoving watermark with Stable Diffusion AI...")
print("This will take 20-30 seconds but produce HitPaw-quality results!\n")

inpainter = SDInpainter(device='cuda')
result = inpainter.inpaint_region(frame, mask)

# Save result
cv2.imwrite('result_sd.jpg', result)
print("\n✅ Watermark removed with Stable Diffusion!")
print("Saved result_sd.jpg - HitPaw-quality removal!")

# Create comparison
comparison = np.hstack([frame, result])
cv2.imwrite('comparison_sd.jpg', comparison)
print("Saved comparison_sd.jpg (before | after)")
