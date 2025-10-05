#!/usr/bin/env python3
"""
Automated watermark removal script using ProPainter
Run this on Windows with: python remove_watermark.py
"""

import os
import sys
import subprocess

# Force pip to install to local watermarkz folder, not C drive
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTALL_DIR = os.path.join(SCRIPT_DIR, 'python_packages')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'pip_cache')
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
os.makedirs(INSTALL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ['PYTHONPATH'] = INSTALL_DIR
os.environ['PIP_CACHE_DIR'] = CACHE_DIR
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
sys.path.insert(0, INSTALL_DIR)

# Check if we're in the right directory
if not os.path.exists('sora_with_watermark.mp4'):
    print("ERROR: sora_with_watermark.mp4 not found!")
    print("Please run this script from D:\\github\\RoomFinderAI\\watermarkz\\")
    sys.exit(1)

# Check if small video exists, if not create it
if not os.path.exists('sora_small.mp4'):
    print("Resizing video to fit 6GB GPU...")
    subprocess.run(['ffmpeg', '-i', 'sora_with_watermark.mp4', '-vf', 'scale=352:640', 'sora_small.mp4'])
    print("Video resized!\n")

VIDEO_FILE = 'sora_small.mp4' if os.path.exists('sora_small.mp4') else 'sora_with_watermark.mp4'

# Check if ProPainter is installed
if not os.path.exists('ProPainter'):
    print("Cloning ProPainter...")
    subprocess.run(['git', 'clone', 'https://github.com/sczhou/ProPainter.git'])

os.chdir('ProPainter')

# Check if requirements are installed
try:
    import torch
    import cv2
    print(f"PyTorch installed: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("\nInstalling dependencies...")
    print("This may take a while...\n")

    # Install PyTorch with CUDA support to watermarkz/python_packages
    print(f"Installing packages to: {INSTALL_DIR}")
    print(f"Using cache directory: {CACHE_DIR}\n")
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        '--target', INSTALL_DIR,
        '--cache-dir', CACHE_DIR,
        '--no-warn-script-location',
        'torch', 'torchvision',
        '--index-url', 'https://download.pytorch.org/whl/cu118'
    ])

    # Install other requirements to watermarkz/python_packages
    subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        '--target', INSTALL_DIR,
        '--cache-dir', CACHE_DIR,
        '--no-warn-script-location',
        '-r', 'requirements.txt'
    ])

    import torch
    import cv2

# Create a mask for the watermark
# For Sora videos, watermark is typically in bottom-right corner
print("\nCreating watermark mask...")
import cv2
import numpy as np

# Read first frame to get dimensions
cap = cv2.VideoCapture(f'../{VIDEO_FILE}')
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read video file!")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"Video dimensions: {w}x{h}")

# Create mask (white = remove, black = keep)
mask = np.zeros((h, w), dtype=np.uint8)

# Sora watermark is in top-middle
watermark_width = int(w * 0.3)  # 30% of width
watermark_height = int(h * 0.08)  # 8% of height
watermark_x = int(w * 0.35)  # Start at 35% from left (centered)
watermark_y = 0  # Top of frame

# Fill watermark area with white
mask[watermark_y:watermark_height, watermark_x:watermark_x+watermark_width] = 255

# Save mask
cv2.imwrite('mask.png', mask)
print(f"Mask created: watermark area at ({watermark_x}, {watermark_y}) - ({w}, {h})")

# Run ProPainter
print("\nRemoving watermark with ProPainter...")
print("This may take several minutes depending on video length and GPU...\n")

fp16_flag = ['--fp16'] if torch.cuda.is_available() else []
memory_flags = [
    '--subvideo_length', '20',  # Process in smaller chunks for 6GB GPU
    '--neighbor_length', '3',   # Reduce neighbor frames
    '--ref_stride', '15'        # Use fewer reference frames
]

cmd = [
    sys.executable,
    'inference_propainter.py',
    '--video', f'../{VIDEO_FILE}',
    '--mask', 'mask.png'
] + fp16_flag + memory_flags

print(f"Running: {' '.join(cmd)}\n")
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✅ SUCCESS! Watermark removed!")
    print("Output saved in: ProPainter/results/")
else:
    print("\n❌ ERROR: ProPainter failed!")
    sys.exit(1)
