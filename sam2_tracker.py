import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import torch
from pathlib import Path

print("Loading SAM2 for watermark tracking...")

try:
    # SAM2 is in IOPaint - try to import
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything import SamAutomaticMaskGenerator

    # Download SAM model if needed
    model_type = "vit_h"
    checkpoint = "sam_vit_h_4b8939.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    print(f"SAM2 loaded on {device}")

except Exception as e:
    print(f"Could not load SAM2: {e}")
    print("\nSAM2 requires manual download.")
    print("Alternative: Use manual point selection for first frame")
    exit()

# Test on first frame
video_path = 'sora_with_watermark.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video!")
    exit()

# User selects watermark point
print("\nClick on the watermark to segment it...")
print("Press SPACE when done")

clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        frame_copy = frame.copy()
        for pt in clicked_points:
            cv2.circle(frame_copy, tuple(pt), 5, (0, 255, 0), -1)
        cv2.imshow('Click watermark - SPACE when done', frame_copy)

cv2.namedWindow('Click watermark - SPACE when done')
cv2.setMouseCallback('Click watermark - SPACE when done', mouse_callback)
cv2.imshow('Click watermark - SPACE when done', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()

if not clicked_points:
    print("No points selected!")
    exit()

# Segment watermark
predictor.set_image(frame)
input_point = np.array(clicked_points)
input_label = np.ones(len(clicked_points))

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Use best mask
best_mask = masks[np.argmax(scores)]

# Save mask
mask_uint8 = (best_mask * 255).astype(np.uint8)
cv2.imwrite('sam2_mask_frame0.png', mask_uint8)

# Show result
result = frame.copy()
result[best_mask] = [0, 255, 0]  # Green overlay
blended = cv2.addWeighted(frame, 0.7, result, 0.3, 0)
cv2.imwrite('sam2_detection.jpg', blended)

print(f"\nSAM2 segmentation saved!")
print("sam2_mask_frame0.png - mask for frame 0")
print("sam2_detection.jpg - preview")
print("\nNow use this as template for tracking!")
