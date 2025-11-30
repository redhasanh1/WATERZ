"""
Expand 10fps masks to physical files for ProPainter
Reads mask_mapping.json and duplicates masks to match original frame count
"""
import os
import cv2
import json
import shutil
from tqdm import tqdm

MASKS_10FPS = r"D:\watermarkz\temp_sam2_masks"
MASKS_EXPANDED = r"D:\watermarkz\temp_sam2_masks_expanded"

print("="*60)
print("EXPAND 10FPS MASKS TO PHYSICAL FILES")
print("="*60)

# Load mapping
mapping_path = os.path.join(MASKS_10FPS, 'mask_mapping.json')
if not os.path.exists(mapping_path):
    print(f"ERROR: No mask_mapping.json found!")
    exit(1)

with open(mapping_path) as f:
    mapping = json.load(f)

print(f"Original FPS: {mapping['original_fps']}")
print(f"Multiplier: {mapping['multiplier']}x")
print(f"10fps masks: {mapping['num_masks']}")
print(f"Target frames: {mapping['total_frames']}")

# Clear/create output dir
if os.path.exists(MASKS_EXPANDED):
    shutil.rmtree(MASKS_EXPANDED)
os.makedirs(MASKS_EXPANDED)

# Load all 10fps masks into memory first
print(f"\nLoading {mapping['num_masks']} masks...")
masks_10fps = []
for i in range(mapping['num_masks']):
    mask_path = os.path.join(MASKS_10FPS, f'{i:05d}.png')
    if os.path.exists(mask_path):
        masks_10fps.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
    else:
        print(f"  WARNING: Missing {mask_path}")
        masks_10fps.append(None)

# Expand and save
print(f"\nExpanding to {mapping['total_frames']} files...")
multiplier = mapping['multiplier']

for frame_idx in tqdm(range(mapping['total_frames']), desc="Saving"):
    mask_idx = min(int(frame_idx / multiplier), len(masks_10fps) - 1)
    mask = masks_10fps[mask_idx]

    if mask is not None:
        out_path = os.path.join(MASKS_EXPANDED, f'{frame_idx:05d}.png')
        cv2.imwrite(out_path, mask)

print(f"\nDONE!")
print(f"Expanded masks: {MASKS_EXPANDED}")
print(f"Files: {mapping['total_frames']}")
print(f"\nNow run START_CELERY_TRT_LOCAL.bat with masks from:")
print(f"  {MASKS_EXPANDED}")
