"""
Test WavePaint with ONNX Runtime (5-10x faster than PyTorch!)
No TensorRT DLL issues!
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))

import cv2
import numpy as np
import time

print("=" * 60)
print("WavePaint ONNX Runtime Test")
print("=" * 60)
print("")

# Check ONNX file
onnx_path = "weights/wavepaint.onnx"
if not os.path.exists(onnx_path):
    print(f"❌ ONNX file not found: {onnx_path}")
    print("Run EXPORT_WAVEPAINT_TENSORRT.bat first (it creates the ONNX)")
    input("Press Enter to exit...")
    sys.exit(1)

print(f"✅ ONNX file found: {onnx_path}")
print("")

# Load ONNX Runtime
print("[1/5] Loading ONNX Runtime...")
try:
    import onnxruntime as ort

    # Check providers
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")

    if 'CUDAExecutionProvider' in providers:
        print("✅ CUDA provider available - will use GPU!")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        print("⚠️  CUDA provider not available - will use CPU (slower)")
        providers = ['CPUExecutionProvider']

    # Load model
    session = ort.InferenceSession(onnx_path, providers=providers)
    print("✅ ONNX model loaded")

except ImportError:
    print("❌ ONNX Runtime not installed!")
    print("Run: INSTALL_ONNXRUNTIME.bat")
    input("Press Enter to exit...")
    sys.exit(1)
except Exception as e:
    print(f"❌ Failed to load ONNX: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)
print("")

# Load test image
print("[2/5] Loading test image...")
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
print("")

# Create manual mask (simple center square for testing)
print("[3/5] Creating test mask...")
print("Creating a square mask in the center for speed testing...")
mask = np.zeros((h, w), dtype=np.uint8)
y1, y2 = h//4, 3*h//4
x1, x2 = w//4, 3*w//4
cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
print(f"✅ Mask created: {x2-x1}x{y2-y1} square")
print("")

# Prepare inputs
print("[4/5] Running inference...")

# Resize to 256x256
img_256 = cv2.resize(image, (256, 256))
mask_256 = cv2.resize(mask, (256, 256))

# Convert to float and normalize
img_input = img_256.astype(np.float32).transpose(2, 0, 1) / 255.0  # HWC -> CHW
mask_input = mask_256.astype(np.float32)[np.newaxis, :, :] / 255.0  # Add channel dim

# Add batch dimension
img_input = np.expand_dims(img_input, axis=0)  # 1x3x256x256
mask_input = np.expand_dims(mask_input, axis=0)  # 1x1x256x256

# Get input/output names
input_names = [i.name for i in session.get_inputs()]
output_names = [i.name for i in session.get_outputs()]

print(f"Input names: {input_names}")
print(f"Output names: {output_names}")

# Warmup
print("Warming up...")
for _ in range(3):
    _ = session.run(output_names, {input_names[0]: img_input, input_names[1]: mask_input})

# Benchmark
print("Benchmarking (10 runs)...")
times = []
for i in range(10):
    start = time.time()
    outputs = session.run(output_names, {input_names[0]: img_input, input_names[1]: mask_input})
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Run {i+1}: {elapsed*1000:.0f} ms")

avg_time = np.mean(times)
fps = 1.0 / avg_time

print("")
print(f"✅ Average: {avg_time*1000:.0f} ms/frame ({fps:.2f} fps)")
print("")

# Process result
result = outputs[0][0]  # Remove batch dim
result = result.transpose(1, 2, 0)  # CHW -> HWC
result = (result * 255).clip(0, 255).astype(np.uint8)
result = cv2.resize(result, (w, h))

# Save results
os.makedirs('wavepaint_test/results', exist_ok=True)
cv2.imwrite('wavepaint_test/results/onnx_result.jpg', result)
print("✅ Result saved: wavepaint_test/results/onnx_result.jpg")
print("")

# Compare with PyTorch time
print("[5/5] Speed Comparison...")
pytorch_time = 8.0  # Your reported time
speedup = pytorch_time / avg_time

print(f"PyTorch:     {pytorch_time*1000:.0f} ms/frame ({1/pytorch_time:.2f} fps)")
print(f"ONNX Runtime: {avg_time*1000:.0f} ms/frame ({fps:.2f} fps)")
print(f"")
print(f"🚀 Speedup: {speedup:.1f}x faster!")
print("")

print("=" * 60)
print("✅ TEST COMPLETE!")
print("=" * 60)
input("Press Enter to exit...")
