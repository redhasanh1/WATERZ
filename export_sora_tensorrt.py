import os
import sys

# Add DLL directory for TensorRT
dll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'python_packages')
os.add_dll_directory(dll_dir)

# Add python_packages to path
sys.path.insert(0, 'python_packages')

import tensorrt as trt
print(f"✅ TensorRT version: {trt.__version__}")

from ultralytics import YOLO

# Find Sora model
sora_model_paths = [
    'runs/detect/sora_watermark/weights/best.pt',
    '../runs/detect/sora_watermark/weights/best.pt',
    '/workspaces/RoomFinderAI/runs/detect/sora_watermark/weights/best.pt',
    'D:/github/RoomFinderAI/runs/detect/sora_watermark/weights/best.pt',
]

sora_model = None
for path in sora_model_paths:
    if os.path.exists(path):
        sora_model = path
        break

if not sora_model:
    print("❌ Sora model not found!")
    print("Run TRAIN_YOLO.bat first to train the Sora watermark detection model")
    sys.exit(1)

print(f"Found Sora model: {sora_model}")
print("\n" + "=" * 60)
print("Exporting to TensorRT Engine (FP16)")
print("=" * 60)
print("This will take 3-5 minutes...")
print()

# Load model
model = YOLO(sora_model)

# Export to TensorRT engine
output_dir = os.path.dirname(sora_model)
engine_path = model.export(
    format='engine',
    device=0,
    half=True,  # FP16 for speed
    imgsz=640,
    workspace=4,  # 4GB workspace
    verbose=True,
    batch=1
)

print(f"\nEngine exported to: {engine_path}")

print("\n" + "=" * 60)
print("✅ EXPORT COMPLETE!")
print("=" * 60)
print(f"TensorRT engine saved to: {engine_path}")
print("\nThis will give you 10-20x speedup!")
print("Restart your processing and it will auto-detect the .engine file")
