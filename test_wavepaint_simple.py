"""
Simple WavePaint Test - Minimal version with error handling
"""
import sys
import os

print("=" * 60)
print("WavePaint Test - Simple Version")
print("=" * 60)
print("")

# Step 1: Check environment
print("[1/8] Checking environment...")
print(f"Python: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print("")

# Step 2: Add paths
print("[2/8] Setting up Python paths...")
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_packages = os.path.join(script_dir, 'python_packages')
    sys.path.insert(0, python_packages)
    sys.path.insert(0, os.path.join(script_dir, 'wavepaint'))
    print(f"✅ Added: {python_packages}")
    print(f"✅ Added: {os.path.join(script_dir, 'wavepaint')}")
except Exception as e:
    print(f"❌ Path setup failed: {e}")
    sys.exit(1)
print("")

# Step 3: Import basic libraries
print("[3/8] Importing libraries...")
try:
    import cv2
    print("✅ OpenCV imported")
except Exception as e:
    print(f"❌ OpenCV import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✅ NumPy imported")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)

try:
    import torch
    print(f"✅ PyTorch imported (version {torch.__version__})")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)
print("")

# Step 4: Create output directories
print("[4/8] Creating output directories...")
try:
    os.makedirs('wavepaint_test/results', exist_ok=True)
    print("✅ Output directories created")
except Exception as e:
    print(f"❌ Failed to create directories: {e}")
    sys.exit(1)
print("")

# Step 5: Load YOLO detector
print("[5/8] Loading YOLO watermark detector...")
try:
    from yolo_detector import YOLOWatermarkDetector
    detector = YOLOWatermarkDetector()
    print("✅ YOLO loaded successfully")
except Exception as e:
    print(f"❌ YOLO failed to load: {e}")
    print("")
    print("Traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("")

# Step 6: Load test image
print("[6/8] Loading test image...")
test_image_path = input("Enter path to test image (or press Enter for comparison.jpg): ").strip()

if not test_image_path:
    test_image_path = "comparison.jpg"

if not os.path.exists(test_image_path):
    print(f"❌ Image not found: {test_image_path}")
    sys.exit(1)

try:
    image = cv2.imread(test_image_path)
    if image is None:
        raise Exception("cv2.imread returned None")
    print(f"✅ Image loaded: {image.shape}")
    cv2.imwrite('wavepaint_test/results/01_original.jpg', image)
except Exception as e:
    print(f"❌ Failed to load image: {e}")
    sys.exit(1)
print("")

# Step 7: Detect watermark
print("[7/8] Detecting watermark with YOLO...")
try:
    detections = detector.detect(image, confidence_threshold=0.25, padding=0)

    if not detections:
        print("⚠️  No watermark detected!")
        print("Trying lower confidence threshold...")
        detections = detector.detect(image, confidence_threshold=0.1, padding=0)

    if not detections:
        print("❌ Still no watermark found. Image may not have a watermark.")
        print("Exiting...")
        sys.exit(0)

    print(f"✅ Found {len(detections)} watermark(s)")
    for i, det in enumerate(detections):
        print(f"   Detection {i+1}: confidence={det['confidence']:.2f}")

    # Create mask
    mask = detector.create_mask(image, detections, padding=10)
    cv2.imwrite('wavepaint_test/results/02_mask.png', mask)
    print("✅ Mask created")

except Exception as e:
    print(f"❌ Detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("")

# Step 8: Check for WavePaint weights
print("[8/8] Checking WavePaint weights...")
available_weights = [
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_thick_mask.pth", "CelebHQ Thick"),
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_medium_mask.pth", "CelebHQ Medium"),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_thick_mask.pth", "Places256 Thick"),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_medium_mask.pth", "Places256 Medium"),
]

weights_path = None
for path, name in available_weights:
    if os.path.exists(path):
        weights_path = path
        print(f"✅ Found weights: {name}")
        print(f"   Path: {path}")
        break

if not weights_path:
    print("❌ No weights found in weights/ folder")
    print("Expected one of:")
    for path, name in available_weights:
        print(f"   - {path}")
    sys.exit(1)
print("")

print("=" * 60)
print("✅ ALL CHECKS PASSED!")
print("=" * 60)
print("")
print("Test image loaded and watermark detected successfully.")
print("WavePaint weights found.")
print("")
print("Next step: Run full inpainting test")
print("(Not implemented yet - this was just a connectivity test)")
print("")
