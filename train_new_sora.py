import sys
sys.path.insert(0, 'python_packages')

from ultralytics import YOLO

print("=" * 60)
print("Training YOLOv8 on NEW Sora Watermarks")
print("=" * 60)

# Load pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='new_sora_dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='new_sora_watermark',
    project='runs/detect',
    device=0,
    patience=10,
    exist_ok=True,
    workers=0  # Fix Windows DataLoader crash
)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nModel saved to: runs/detect/new_sora_watermark/weights/best.pt")
print("\nNext step: Export to TensorRT using EXPORT_NEW_SORA_TENSORRT.bat")
