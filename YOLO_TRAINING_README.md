# YOLOv8 Sora Watermark Detection Training

## Overview
Train YOLOv8 to specifically detect Sora watermarks (the ghost icon + "Sora" text).

## Why Train YOLOv8?
The generic YOLOv8 watermark detector doesn't know what Sora's watermark looks like. By training it on Sora-specific examples, it will learn to detect that exact pattern.

## Training Steps (Run in order on Windows)

### Step 1: Extract Training Frames
```
EXTRACT_FRAMES.bat
```
This extracts 40 frames from `sora_with_watermark.mp4` into `yolo_training/images/`

### Step 2: Create YOLO Labels
```
CREATE_LABELS.bat
```
This automatically detects the Sora watermark in each frame using template matching and creates YOLO annotation files in `yolo_training/labels/`

### Step 3: Train YOLOv8
```
TRAIN_YOLO.bat
```
This trains YOLOv8 for 100 epochs. Takes 5-10 minutes.

Training output saved to: `runs/detect/sora_watermark/weights/best.pt`

### Step 4: Test Detection
```
TEST_YOLO.bat
```
Tests the trained model on `sora_test_frame.jpg`

## How It Works

1. **Dataset Creation**: Extract frames and use template matching to auto-label the watermark locations
2. **Training**: Fine-tune YOLOv8n (nano) on the labeled Sora watermark dataset
3. **Detection**: The trained model can now detect Sora watermarks with high accuracy

## Trained Model Usage

After training, `yolo_detector.py` will automatically use the trained model:

```python
from yolo_detector import YOLOWatermarkDetector

detector = YOLOWatermarkDetector()  # Auto-loads trained Sora model
detections = detector.detect(frame, confidence_threshold=0.5)
```

## Files Created

- `yolo_training/images/*.jpg` - Training images
- `yolo_training/labels/*.txt` - YOLO format labels
- `runs/detect/sora_watermark/weights/best.pt` - Trained model
- `runs/detect/sora_watermark/` - Training metrics and plots

## Troubleshooting

**No watermarks detected after training:**
- Lower confidence threshold: `detector.detect(frame, confidence_threshold=0.3)`
- Train for more epochs: Edit `train_yolo_sora.py` line with `epochs=100` to `epochs=200`

**Training fails:**
- Check that CUDA/GPU is available or set `device='cpu'` in `train_yolo_sora.py`
- Reduce batch size: Change `batch=8` to `batch=4` or `batch=2`

**Want to retrain:**
- Delete `runs/detect/sora_watermark/` folder
- Run `TRAIN_YOLO.bat` again
