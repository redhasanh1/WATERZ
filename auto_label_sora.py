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
print("2. Press 'Enter' to save and go to NEXT image")
print("3. Press 'S' to skip this image (no watermark)")
print("4. Press 'R' to redraw current image")
print("5. Press 'Q' to quit")
print()

cv2.namedWindow('Label Sora Watermark', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Label Sora Watermark', 1000, 700)
cv2.setMouseCallback('Label Sora Watermark', draw_rectangle)

# Label each image
labeled_count = 0

for idx, img_path in enumerate(image_files):
    print(f"\n[{idx+1}/{len(image_files)}] {os.path.basename(img_path)}")

    current_image = cv2.imread(img_path)
    if current_image is None:
        print("  ❌ Failed to load, skipping...")
        continue

    h, w = current_image.shape[:2]

    # Resize to fit screen (max 1000 width)
    max_width = 1000
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    display_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Store original dimensions for YOLO labels
    orig_h, orig_w = h, w
    h, w = display_image.shape[:2]

    bbox = None
    first_image = display_image.copy()

    cv2.imshow('Label Sora Watermark', display_image)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter - save and next
            if bbox:
                # Convert bbox to YOLO format (use ORIGINAL image dimensions)
                x1, y1, x2, y2 = bbox
                x_center = ((x1 + x2) / 2) / orig_w
                y_center = ((y1 + y2) / 2) / orig_h
                width_norm = (x2 - x1) / orig_w
                height_norm = (y2 - y1) / orig_h

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
