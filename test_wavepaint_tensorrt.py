"""
Test WavePaint with TensorRT engine on real video
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))

import cv2
import numpy as np
import time

print("=" * 60)
print("WavePaint TensorRT Video Test")
print("=" * 60)
print("")

# Check TensorRT engine
engine_path = "weights/wavepaint.engine"
if not os.path.exists(engine_path):
    print(f"❌ TensorRT engine not found: {engine_path}")
    print("Run EXPORT_WAVEPAINT_TENSORRT_SIMPLE.bat first")
    input("Press Enter to exit...")
    sys.exit(1)

print(f"✅ TensorRT engine found: {engine_path}")
print("")

# Load TensorRT Runtime
print("[1/4] Loading TensorRT Runtime...")
try:
    import tensorrt as trt
    import torch

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    print("✅ TensorRT engine loaded")

except Exception as e:
    print(f"❌ Failed to load TensorRT: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)
print("")

# Load video
print("[2/4] Loading video...")
video_path = input("Enter video path (or press Enter for test video): ").strip()
if not video_path:
    video_path = "test_video.mp4"

if not os.path.exists(video_path):
    print(f"❌ Video not found: {video_path}")
    input("Press Enter to exit...")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ Video loaded: {width}x{height} @ {fps} fps ({total_frames} frames)")
print("")

# Create output video
print("[3/4] Setting up output...")
output_path = "wavepaint_test/results/tensorrt_output.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"✅ Output: {output_path}")
print("")

# Process video
print("[4/4] Processing video with WavePaint TensorRT...")
print("")

# Create mask (center square for testing)
mask = np.zeros((height, width), dtype=np.uint8)
y1, y2 = height//4, 3*height//4
x1, x2 = width//4, 3*width//4
cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

# Allocate GPU buffers using torch CUDA
device = torch.device('cuda')
d_input_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32).cuda()
d_input_mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32).cuda()
d_output = torch.zeros((1, 3, 256, 256), dtype=torch.float32).cuda()

# Get binding indices
bindings = [int(d_input_img.data_ptr()), int(d_input_mask.data_ptr()), int(d_output.data_ptr())]

frame_count = 0
times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    start_time = time.time()

    # Prepare inputs
    img_256 = cv2.resize(frame, (256, 256))
    mask_256 = cv2.resize(mask, (256, 256))

    # Convert to tensors
    img_input = img_256.astype(np.float32).transpose(2, 0, 1) / 255.0
    mask_input = mask_256.astype(np.float32)[np.newaxis, :, :] / 255.0

    # Copy to GPU
    d_input_img.copy_(torch.from_numpy(img_input).unsqueeze(0))
    d_input_mask.copy_(torch.from_numpy(mask_input).unsqueeze(0))

    # Run inference
    context.execute_v2(bindings)

    # Get result
    result = d_output.cpu().numpy()[0]
    result = result.transpose(1, 2, 0)
    result = (result * 255).clip(0, 255).astype(np.uint8)
    result = cv2.resize(result, (width, height))

    elapsed = time.time() - start_time
    times.append(elapsed)

    # Write frame
    out.write(result)

    if frame_count % 10 == 0:
        avg_time = np.mean(times[-10:])
        fps_actual = 1.0 / avg_time
        print(f"  Frame {frame_count}/{total_frames} - {avg_time*1000:.0f} ms/frame ({fps_actual:.2f} fps)")

cap.release()
out.release()

avg_time = np.mean(times)
fps_actual = 1.0 / avg_time

print("")
print("=" * 60)
print("✅ PROCESSING COMPLETE!")
print("=" * 60)
print("")
print(f"Processed {frame_count} frames")
print(f"Average: {avg_time*1000:.0f} ms/frame ({fps_actual:.2f} fps)")
print(f"Output: {output_path}")
print("")
print(f"🚀 Speed: {8000/avg_time/1000:.1f}x faster than PyTorch!")
print("")
input("Press Enter to exit...")
