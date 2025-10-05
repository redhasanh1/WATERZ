import sys
sys.path.insert(0, 'python_packages')

import cv2
import os
import glob

# Sora watermark is always at the top-center
# From analyzing sora_test_frame.jpg, the watermark is approximately:
# - Width: ~150 pixels (ghost icon + "Sora" text)
# - Height: ~40 pixels
# - Position: top-center of frame

template = cv2.imread('watermark_template.png')
if template is None:
    print("Error: watermark_template.png not found!")
    exit()

th, tw = template.shape[:2]
print(f"Watermark template size: {tw}x{th}")

# Create labels directory
os.makedirs('yolo_training/labels', exist_ok=True)

image_files = sorted(glob.glob('yolo_training/images/*.jpg'))

if not image_files:
    print("No images found! Run extract_training_frames.py first")
    exit()

print(f"\nCreating YOLO labels for {len(image_files)} images...\n")

for img_path in image_files:
    # Read image
    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]

    # Use template matching to find watermark
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Get bounding box
    x1, y1 = max_loc
    x2 = x1 + tw
    y2 = y1 + th

    # Add padding (20 pixels on each side)
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Convert to YOLO format (normalized center coordinates + width/height)
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    # YOLO label format: class_id x_center y_center width height
    # class_id = 0 (sora_watermark is our only class)
    label_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # Save label file
    label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    with open(label_path, 'w') as f:
        f.write(label_line)

    filename = os.path.basename(img_path)
    print(f"{filename}: bbox=({x1},{y1})-({x2},{y2}) confidence={max_val:.2%}")

print(f"\n✅ Created {len(image_files)} label files in yolo_training/labels/")
print("\nNext step: Run train_yolo_sora.py to train the model")
