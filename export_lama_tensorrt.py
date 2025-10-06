import os
import sys

# CRITICAL: Force everything to stay in watermarkz folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # Make sure we're in watermarkz folder

# Set all temp/cache to watermarkz folder ONLY
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

# Add python_packages to path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'python_packages'))

import torch
import cv2
import numpy as np

print("=" * 60)
print("WORKING DIRECTORY CHECK")
print("=" * 60)
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {SCRIPT_DIR}")
print(f"Temp directory: {TEMP_DIR}")
print(f"Cache directory: {CACHE_DIR}")
print("\n✅ Everything will stay in watermarkz folder!")
print("=" * 60)

print("=" * 60)
print("LAMA TENSORRT EXPORT")
print("=" * 60)

# Check CUDA
if not torch.cuda.is_available():
    print("❌ CUDA not available! TensorRT requires CUDA.")
    sys.exit(1)

print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")

# Load LAMA model
print("\nLoading LAMA model...")
from iopaint.model.lama import LaMa

device = torch.device('cuda')
lama = LaMa(device)

print("✅ LAMA model loaded")

# Get the actual PyTorch model (TorchScript JIT model)
model = lama.model

print(f"\nModel type: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")

# Create dummy input for tracing
print("\nCreating dummy input for export...")
dummy_image = torch.randn(1, 3, 512, 512).cuda()
dummy_mask = torch.randn(1, 1, 512, 512).cuda()

print(f"  Image shape: {dummy_image.shape}")
print(f"  Mask shape: {dummy_mask.shape}")

# Export to ONNX (inside watermarkz folder)
onnx_path = os.path.join(SCRIPT_DIR, 'lama_inpainting.onnx')
print(f"\n" + "=" * 60)
print(f"Exporting to ONNX: {onnx_path}")
print("=" * 60)

try:
    # Set model to eval mode
    model.eval()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_image, dummy_mask),
            onnx_path,
            export_params=True,
            opset_version=17,  # Use latest ONNX opset
            do_constant_folding=True,
            input_names=['image', 'mask'],
            output_names=['output'],
            dynamic_axes={
                'image': {0: 'batch', 2: 'height', 3: 'width'},
                'mask': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch', 2: 'height', 3: 'width'}
            }
        )

    print(f"✅ ONNX export successful: {onnx_path}")

except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Convert ONNX to TensorRT
print(f"\n" + "=" * 60)
print("Converting ONNX to TensorRT Engine")
print("=" * 60)
print("\nThis will take 5-10 minutes...")
print("TensorRT is optimizing the model for your GPU...")

engine_path = os.path.join(SCRIPT_DIR, 'lama_inpainting.engine')

# Use trtexec command to build TensorRT engine
trtexec_cmd = f"""
trtexec \\
    --onnx={onnx_path} \\
    --saveEngine={engine_path} \\
    --fp16 \\
    --workspace=4096 \\
    --minShapes=image:1x3x256x256,mask:1x1x256x256 \\
    --optShapes=image:1x3x512x512,mask:1x1x512x512 \\
    --maxShapes=image:1x3x1024x1024,mask:1x1x1024x1024 \\
    --verbose
"""

print(f"\nRunning trtexec command:")
print(trtexec_cmd)

import subprocess
result = subprocess.run(
    trtexec_cmd.replace('\n', ' ').replace('\\', ''),
    shell=True,
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print(f"\n✅ TensorRT engine created: {engine_path}")
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nTensorRT engine saved to: {engine_path}")
    print("\nNext steps:")
    print("1. Update lama_inpaint.py to use this .engine file")
    print("2. Expect 2-5x faster inpainting!")
    print("=" * 60)
else:
    print(f"\n❌ TensorRT conversion failed!")
    print(f"\nError output:")
    print(result.stderr)
    print("\n⚠️ Note: trtexec must be in PATH")
    print("   TensorRT location: TensorRT-10.7.0.23/bin/")
