import sys
sys.path.insert(0, 'python_packages')

from ultralytics import YOLO
import os

def train_sora_model():
    print("=" * 60)
    print("Training YOLOv8 on Sora Watermark Dataset")
    print("=" * 60)

    # Check if dataset exists
    if not os.path.exists('yolo_training/images'):
        print("\n❌ Error: Training images not found!")
        print("Run EXTRACT_FRAMES.bat first")
        exit()

    if not os.path.exists('yolo_training/labels'):
        print("\n❌ Error: Training labels not found!")
        print("Run CREATE_LABELS.bat first")
        exit()

    # Count training samples
    num_images = len([f for f in os.listdir('yolo_training/images') if f.endswith('.jpg')])
    num_labels = len([f for f in os.listdir('yolo_training/labels') if f.endswith('.txt')])

    print(f"\n📊 Dataset:")
    print(f"   Images: {num_images}")
    print(f"   Labels: {num_labels}")

    if num_images != num_labels:
        print("\n⚠️  Warning: Image and label count mismatch!")

    # Load pre-trained YOLOv8n model (nano - fastest)
    print("\n🔄 Loading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')  # Start from YOLOv8 nano

    # Train the model
    print("\n🚀 Starting training...")
    print("This will take 5-10 minutes depending on your GPU/CPU\n")

    results = model.train(
        data='sora_dataset.yaml',     # dataset config
        epochs=100,                   # train for 100 epochs
        imgsz=640,                    # image size
        batch=8,                      # batch size (adjust based on GPU memory)
        name='sora_watermark',        # experiment name
        patience=20,                  # early stopping patience
        save=True,                    # save checkpoints
        plots=True,                   # save training plots
        device=0,                     # use GPU 0 (change to 'cpu' if no GPU)
        exist_ok=True,                # overwrite existing experiment
        workers=0,                    # Fix Windows multiprocessing issue
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Model saved to: runs/detect/sora_watermark/weights/best.pt")
    print("\nNext: Run TEST_YOLO.bat to test the trained model")

if __name__ == '__main__':
    train_sora_model()
