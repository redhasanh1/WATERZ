#!/bin/bash
# ============================================================
# Build YOLO Sora v2 TensorRT Engine for WSL
# ============================================================
# Output goes to separate folder to avoid overwriting Windows engine
# ============================================================

set -e

cd /mnt/d/watermarkz
source venv_wsl2/bin/activate

echo ""
echo "============================================================"
echo "Building YOLO Sora v2 TensorRT Engine (WSL/Linux)"
echo "============================================================"
echo ""

# Source model
PT_MODEL="runs/detect/sora_watermark_v2/weights/best.pt"

# Output to separate WSL folder
OUTPUT_DIR="runs/detect/sora_watermark_v2/weights/wsl"
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$PT_MODEL" ]; then
    echo "[ERROR] Model not found: $PT_MODEL"
    exit 1
fi

echo "[INFO] Source model: $PT_MODEL"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo ""
echo "Building FP16 batch=64 engine (this takes 3-5 minutes)..."
echo ""

python3 << 'EOF'
import os
import shutil
from ultralytics import YOLO

pt_model = "runs/detect/sora_watermark_v2/weights/best.pt"
output_dir = "runs/detect/sora_watermark_v2/weights/wsl"

print(f"[YOLO] Loading model: {pt_model}")
model = YOLO(pt_model)

print(f"[YOLO] Exporting to TensorRT FP16 (batch=64)...")
engine_path = model.export(
    format='engine',
    device=0,
    half=True,
    imgsz=640,
    batch=64,
    workspace=4
)

# Move to WSL subfolder with clear name
final_path = os.path.join(output_dir, "best_fp16_batch64_wsl.engine")
shutil.move(engine_path, final_path)

print(f"")
print(f"[SUCCESS] Engine built: {final_path}")
print(f"[SIZE] {os.path.getsize(final_path) / 1024 / 1024:.1f} MB")
EOF

echo ""
echo "============================================================"
echo "DONE! Update wsl_sam2_worker.py line 291 to:"
echo "  trt_engine_path = '/mnt/d/watermarkz/runs/detect/sora_watermark_v2/weights/wsl/best_fp16_batch64_wsl.engine'"
echo "============================================================"
