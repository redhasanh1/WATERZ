import sys
sys.path.insert(0, 'python_packages')

import cv2
import os
import glob

print("=" * 60)
print("Auto-Label Sora Watermarks")
print("=" * 60)
print("\nThis will help you label the Sora watermark position.")
print("Click and drag to draw a box around the watermark on the first frame.")
print()

images_dir = 'NEW_SORA_TRAINING/images'
labels_dir = 'NEW_SORA_TRAINING/labels'

os.makedirs(labels_dir, exist_ok=True)

# Get all images
image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))

if not image_files:
    print(f"❌ No images found in {images_dir}/")
    print("Run EXTRACT_FRAMES.bat first!")
    exit()

print(f"Found {len(image_files)} images")

# Load first image
first_image = cv2.imread(image_files[0])
if first_image is None:
    print("Error loading first image!")
    exit()

h, w = first_image.shape[:2]
print(f"Image size: {w}x{h}")

# Variables for drawing
drawing = False
bbox = None
ix, iy = -1, -1

# Mouse callback
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox, first_image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = first_image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Label Sora Watermark', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
        img_copy = first_image.copy()
        cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.imshow('Label Sora Watermark', img_copy)

print("\nInstructions:")
print("1. Draw a box around the Sora watermark")
print("2. Press 'SPACEBAR' to save and go to NEXT image")
print("3. Press 'S' to skip this image (no watermark)")
print("4. Press 'R' to redraw current image")
print("5. Press 'Q' to quit")
print()

cv2.namedWindow('Label Sora Watermark', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Label Sora Watermark', draw_rectangle)

# Label each image
labeled_count = 0

for idx, img_path in enumerate(image_files):
    img_name = os.path.basename(img_path)
    label_name = os.path.splitext(img_name)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_name)

    # Skip if already labeled
    if os.path.exists(label_path):
        print(f"\n[{idx+1}/{len(image_files)}] {img_name} - ✅ Already labeled, skipping...")
        continue

    print(f"\n[{idx+1}/{len(image_files)}] {img_name}")

    current_image = cv2.imread(img_path)
    if current_image is None:
        print("  ❌ Failed to load, skipping...")
        continue

    orig_h, orig_w = current_image.shape[:2]

    # Resize to fit screen (max 600 width to fit on screen)
    max_width = 600
    scale = max_width / orig_w
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    display_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w = display_image.shape[:2]
    scale = w / orig_w  # Scale factor for converting bbox back to original

    bbox = None
    first_image = display_image.copy()

    cv2.imshow('Label Sora Watermark', display_image)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # Spacebar - save and next
            if bbox:
                # Convert bbox from display coords to original coords, then YOLO format
                x1, y1, x2, y2 = bbox

                # Scale back to original image coordinates
                x1_orig = x1 / scale
                y1_orig = y1 / scale
                x2_orig = x2 / scale
                y2_orig = y2 / scale

                # Convert to YOLO format (normalized 0-1)
                x_center = ((x1_orig + x2_orig) / 2) / orig_w
                y_center = ((y1_orig + y2_orig) / 2) / orig_h
                width_norm = (x2_orig - x1_orig) / orig_w
                height_norm = (y2_orig - y1_orig) / orig_h

                # Save label
                img_name = os.path.basename(img_path)
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)

                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

                print(f"  ✅ Saved label: ({x1}, {y1}) to ({x2}, {y2})")
                labeled_count += 1
                break
            else:
                print("  ⚠️  Draw a box first or press 'S' to skip!")

        elif key == ord('s'):  # Skip
            print("  ⏭️  Skipped (no watermark)")
            break

        elif key == ord('r'):  # Redraw
            bbox = None
            first_image = display_image.copy()
            cv2.imshow('Label Sora Watermark', display_image)
            print("  🔄 Redraw...")

        elif key == ord('q'):  # Quit
            print("\n❌ Labeling cancelled")
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()

print(f"\n✅ Labeled {labeled_count} images out of {len(image_files)}")
print(f"Labels saved to {labels_dir}/")
print("\nNext step: Run TRAIN_NEW_SORA.bat to train the model!")
