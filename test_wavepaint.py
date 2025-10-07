"""
Test WavePaint Inpainting vs LaMa
Uses YOLO for detection, compares WavePaint and LaMa for inpainting
"""
import sys
import os

# Add python_packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))

import cv2
import numpy as np
import torch
from pathlib import Path

# Create output directories
os.makedirs('wavepaint_test/results', exist_ok=True)
os.makedirs('wavepaint_test/masks', exist_ok=True)

print("=" * 60)
print("WavePaint vs LaMa Comparison Test")
print("=" * 60)

# Step 1: Load YOLO detector
print("\n[1/6] Loading YOLO watermark detector...")
try:
    from yolo_detector import YOLOWatermarkDetector
    detector = YOLOWatermarkDetector()
    print("✅ YOLO loaded successfully")
except Exception as e:
    print(f"❌ Failed to load YOLO: {e}")
    sys.exit(1)

# Step 2: Load test image
print("\n[2/6] Loading test image...")
test_image_path = input("Enter path to test image (or press Enter for default): ").strip()

if not test_image_path:
    # Look for test images in current directory
    test_images = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png'))
    if test_images:
        test_image_path = str(test_images[0])
        print(f"Using: {test_image_path}")
    else:
        print("❌ No test image found. Please provide a path.")
        sys.exit(1)

image = cv2.imread(test_image_path)
if image is None:
    print(f"❌ Failed to load image: {test_image_path}")
    sys.exit(1)

print(f"✅ Image loaded: {image.shape}")
cv2.imwrite('wavepaint_test/results/01_original.jpg', image)

# Step 3: Detect watermark with YOLO
print("\n[3/6] Detecting watermark with YOLO...")
detections = detector.detect(image, confidence_threshold=0.25, padding=0)

if not detections:
    print("⚠️  No watermark detected!")
    print("Image might not have a watermark, or confidence is too low.")
    sys.exit(0)

print(f"✅ Found {len(detections)} watermark(s)")
for i, det in enumerate(detections):
    print(f"   Detection {i+1}: confidence={det['confidence']:.2f}, bbox={det['bbox']}")

# Step 4: Create mask from detections
print("\n[4/6] Creating mask from detections...")
mask = detector.create_mask(image, detections, padding=10)
cv2.imwrite('wavepaint_test/results/02_mask.png', mask)
print("✅ Mask created and saved")

# Step 5: Test LaMa (current method)
print("\n[5/6] Testing LaMa inpainting...")
try:
    from lama_inpaint_optimized import LamaInpainterOptimized
    lama = LamaInpainterOptimized()
    lama_result = lama.inpaint_region(image, mask)
    cv2.imwrite('wavepaint_test/results/03_lama_result.jpg', lama_result)
    print("✅ LaMa inpainting complete")
except Exception as e:
    print(f"⚠️  LaMa failed: {e}")
    print("Trying standard LaMa...")
    try:
        from lama_inpaint_local import LamaInpainter
        lama = LamaInpainter()
        lama_result = lama.inpaint_region(image, mask)
        cv2.imwrite('wavepaint_test/results/03_lama_result.jpg', lama_result)
        print("✅ LaMa inpainting complete")
    except Exception as e2:
        print(f"❌ Both LaMa versions failed: {e2}")
        lama_result = None

# Step 6: Test WavePaint
print("\n[6/6] Testing WavePaint inpainting...")
print("⚠️  NOTE: WavePaint requires pretrained weights!")
print("Download from: https://github.com/pranavphoenix/WavePaint/releases")
print("")

# Check available weights
available_weights = [
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_thick_mask.pth", "CelebHQ Thick Mask"),
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_medium_mask.pth", "CelebHQ Medium Mask"),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_thick_mask.pth", "Places256 Thick Mask"),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_medium_mask.pth", "Places256 Medium Mask"),
]

# Find first available weights
weights_path = None
weights_name = None
for path, name in available_weights:
    if os.path.exists(path):
        weights_path = path
        weights_name = name
        break

if not weights_path:
    print(f"❌ No weights found in weights/ folder")
    print("Please download WavePaint weights from Hugging Face")
    print("")
    print("For now, skipping WavePaint test...")
    wavepaint_result = None
else:
    print(f"✅ Using weights: {weights_name}")
    print(f"   Path: {weights_path}")
    print("")
    try:
        # Import WavePaint model
        sys.path.insert(0, 'wavepaint')
        from model import WavePaint

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize model
        model = WavePaint(
            num_modules=8,
            blocks_per_module=4,
            mult=4,
            ff_channel=128,
            final_dim=128,
            dropout=0.5
        ).to(device)

        # Load weights
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print("✅ WavePaint model loaded")

        # Prepare image and mask for WavePaint
        # WavePaint expects normalized tensors [0, 1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # Resize to 256x256 (WavePaint's training size)
        h, w = image.shape[:2]
        img_256 = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=(256, 256),
            mode='bilinear'
        )
        mask_256 = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),
            size=(256, 256),
            mode='nearest'
        )

        # Apply mask (masked regions = 0)
        masked_img = img_256 * (1 - mask_256)

        # Run inference
        with torch.no_grad():
            output = model(masked_img.to(device), mask_256.to(device))

        # Convert back to image
        result_256 = output[0].permute(1, 2, 0).cpu().numpy() * 255
        result_256 = result_256.astype(np.uint8)

        # Resize back to original size
        wavepaint_result = cv2.resize(result_256, (w, h))

        cv2.imwrite('wavepaint_test/results/04_wavepaint_result.jpg', wavepaint_result)
        print("✅ WavePaint inpainting complete")

    except Exception as e:
        print(f"❌ WavePaint failed: {e}")
        import traceback
        traceback.print_exc()
        wavepaint_result = None

# Summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"Test image: {test_image_path}")
print(f"Watermarks detected: {len(detections)}")
print(f"")
print("Output files saved to: wavepaint_test/results/")
print("  01_original.jpg      - Original image with watermark")
print("  02_mask.png          - YOLO detection mask")
if lama_result is not None:
    print("  03_lama_result.jpg   - LaMa inpainting result")
if wavepaint_result is not None:
    print("  04_wavepaint_result.jpg - WavePaint inpainting result")
print("")

if wavepaint_result is not None and lama_result is not None:
    print("✅ Both methods completed! Compare the results visually.")
elif wavepaint_result is None and lama_result is not None:
    print("⚠️  Only LaMa completed. Download WavePaint weights to compare.")
elif wavepaint_result is not None and lama_result is None:
    print("⚠️  Only WavePaint completed.")
else:
    print("❌ Both methods failed. Check errors above.")

print("=" * 60)
