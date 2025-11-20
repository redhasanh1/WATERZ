#!/bin/bash
set -e
cd /mnt/d/watermarkz

echo "🔧 Installing WSL2 dependencies..."
source venv_wsl2/bin/activate

# Install ultralytics (YOLO)
echo "📦 Installing ultralytics..."
pip install -q ultralytics

# Check if TensorRT engine exists
ENGINE_PATH="runs/detect/new_sora_watermark/weights/best_fp16_batch_rtx4090.engine"
if [ ! -f "$ENGINE_PATH" ]; then
    echo "⚠️  TensorRT engine not found at $ENGINE_PATH"
    echo "💡 Will use regular YOLO (slower but works)"
else
    echo "✅ TensorRT engine found: $ENGINE_PATH"
fi

echo "✅ WSL2 setup complete!"
