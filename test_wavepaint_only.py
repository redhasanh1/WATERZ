"""
Test ONLY WavePaint - No YOLO, No TensorRT bullshit
Just pure inpainting comparison: LaMa vs WavePaint
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wavepaint'))

import cv2
import numpy as np
import torch

print("=" * 60)
print("WavePaint Inpainting Test (No YOLO)")
print("=" * 60)
print("")

# Create output directory
os.makedirs('wavepaint_test/results', exist_ok=True)

# Load test image
print("[1/5] Loading test image...")
test_image = input("Enter image path (or press Enter for weights/test_image.jpg): ").strip()
if not test_image:
    test_image = "weights/test_image.jpg"

if not os.path.exists(test_image):
    print(f"❌ Image not found: {test_image}")
    input("Press Enter to exit...")
    sys.exit(1)

image = cv2.imread(test_image)
h, w = image.shape[:2]
print(f"✅ Image loaded: {w}x{h}")
cv2.imwrite('wavepaint_test/results/01_original.jpg', image)
print("")

# Create manual mask
print("[2/5] Creating mask...")
print("Draw a rectangle on the watermark area")
print("  Left click and drag to draw")
print("  Press 'r' to reset")
print("  Press 'c' to continue")
print("")

clone = image.copy()
mask = np.zeros((h, w), dtype=np.uint8)
drawing = False
ix, iy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, clone, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = clone.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('Draw Watermark Area', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(clone, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.rectangle(mask, (ix, iy), (x, y), 255, -1)
        cv2.imshow('Draw Watermark Area', clone)

cv2.namedWindow('Draw Watermark Area')
cv2.setMouseCallback('Draw Watermark Area', draw_rectangle)
cv2.imshow('Draw Watermark Area', clone)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Continue
        break
    elif key == ord('r'):  # Reset
        clone = image.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imshow('Draw Watermark Area', clone)

cv2.destroyAllWindows()

if mask.sum() == 0:
    print("❌ No mask drawn! Please draw a rectangle over the watermark.")
    input("Press Enter to exit...")
    sys.exit(1)

cv2.imwrite('wavepaint_test/results/02_mask.png', mask)
print("✅ Mask created")
print("")

# Test LaMa
print("[3/5] Testing LaMa inpainting...")
try:
    from lama_inpaint_optimized import LamaInpainterOptimized
    lama = LamaInpainterOptimized()
    print("✅ Using optimized LaMa")
except:
    try:
        from lama_inpaint_local import LamaInpainter
        lama = LamaInpainter()
        print("✅ Using standard LaMa")
    except Exception as e:
        print(f"❌ LaMa failed: {e}")
        lama = None

if lama:
    lama_result = lama.inpaint_region(image, mask)
    cv2.imwrite('wavepaint_test/results/03_lama_result.jpg', lama_result)
    print("✅ LaMa complete")
else:
    print("⚠️  Skipping LaMa")
print("")

# Test WavePaint
print("[4/5] Testing WavePaint inpainting...")

# Find weights
weights = [
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_thick_mask.pth", "CelebHQ Thick"),
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_medium_mask.pth", "CelebHQ Medium"),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_thick_mask.pth", "Places256 Thick"),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_medium_mask.pth", "Places256 Medium"),
]

weights_path = None
for path, name in weights:
    if os.path.exists(path):
        weights_path = path
        print(f"✅ Using: {name}")
        break

if not weights_path:
    print("❌ No WavePaint weights found in weights/ folder")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    from model import WavePaint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Load model with correct parameters for the checkpoint
    # The weights were trained with mult=2, not mult=4
    model = WavePaint(
        num_modules=8,
        blocks_per_module=4,
        mult=2,  # Changed from 4 to 2 to match checkpoint
        ff_channel=128,
        final_dim=128,
        dropout=0.5
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✅ Model loaded")

    # Prepare tensors
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

    # Resize to 256x256 (WavePaint training size)
    img_256 = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0), size=(256, 256), mode='bilinear'
    )
    mask_256 = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0), size=(256, 256), mode='nearest'
    )

    # Apply mask
    masked_img = img_256 * (1 - mask_256)

    # Inpaint
    with torch.no_grad():
        output = model(masked_img.to(device), mask_256.to(device))

    # Convert back
    result_256 = output[0].permute(1, 2, 0).cpu().numpy() * 255
    result_256 = np.clip(result_256, 0, 255).astype(np.uint8)

    # Resize back to original size
    wavepaint_result = cv2.resize(result_256, (w, h))

    cv2.imwrite('wavepaint_test/results/04_wavepaint_result.jpg', wavepaint_result)
    print("✅ WavePaint complete")

except Exception as e:
    print(f"❌ WavePaint failed: {e}")
    import traceback
    traceback.print_exc()
print("")

# Show results
print("[5/5] Results saved!")
print("=" * 60)
print("Check: wavepaint_test/results/")
print("  01_original.jpg      - Original image")
print("  02_mask.png          - Your mask")
if lama:
    print("  03_lama_result.jpg   - LaMa result")
print("  04_wavepaint_result.jpg - WavePaint result")
print("=" * 60)
print("")
input("Press Enter to exit...")
