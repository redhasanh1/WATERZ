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

model_path = 'runs/detect/new_sora_watermark/weights/best.pt'

if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("Run TRAIN_NEW_SORA.bat first!")
    sys.exit(1)

print(f"Found model: {model_path}")
print("\n" + "=" * 60)
print("Exporting to TensorRT Engine (FP16)")
print("=" * 60)
print("This will take 3-5 minutes...")
print()

# Load model
model = YOLO(model_path)

# Export to TensorRT engine
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
print(f"TensorRT engine: {engine_path}")
print("\nNow update yolo_detector.py to use this new engine!")
print("Or copy it to: runs/detect/sora_watermark/weights/best.engine")
