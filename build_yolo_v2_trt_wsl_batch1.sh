#!/bin/bash
# Build YOLO Sora v2 TensorRT Engine for WSL - BATCH=1 (single frame inference)

set -e
cd /mnt/d/watermarkz
source venv_wsl2/bin/activate

echo ""
echo "============================================================"
echo "Building YOLO Sora v2 TensorRT Engine (WSL/Linux) - BATCH=1"
echo "============================================================"
echo ""

PT_MODEL="runs/detect/sora_watermark_v2/weights/best.pt"
OUTPUT_DIR="runs/detect/sora_watermark_v2/weights/wsl"
mkdir -p "$OUTPUT_DIR"

python3 << 'EOF'
from ultralytics import YOLO
import shutil
import os

model = YOLO("runs/detect/sora_watermark_v2/weights/best.pt")
print("[YOLO] Exporting TensorRT FP16 (batch=1)...")

engine_path = model.export(
    format='engine',
    device=0,
    half=True,
    imgsz=640,
    batch=1,
    workspace=4
)

final_path = "runs/detect/sora_watermark_v2/weights/wsl/best_fp16_batch1_wsl.engine"
shutil.move(engine_path, final_path)
print(f"[OK] Engine: {final_path}")
print(f"[SIZE] {os.path.getsize(final_path) / 1024 / 1024:.1f} MB")
EOF

echo ""
echo "DONE! Update wsl_sam2_worker.py to use: best_fp16_batch1_wsl.engine"
