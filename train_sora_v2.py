import sys
sys.path.insert(0, 'python_packages')

from ultralytics import YOLO

print("=" * 60)
print("Training YOLOv8 on Sora Watermarks v2")
print("Dataset: yolo_training (4,481 frames)")
print("=" * 60)

# Load pretrained YOLOv8n model (nano - fastest)
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='sora_dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,  # RTX 4090
    name='sora_watermark_v2',
    project='runs/detect',
    device=0,
    patience=10,
    exist_ok=True,
    workers=0  # Fix Windows DataLoader crash
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print("\nModel saved to: runs/detect/sora_watermark_v2/weights/best.pt")
print("\nNext step: Run export_sora_v2_tensorrt.py")
