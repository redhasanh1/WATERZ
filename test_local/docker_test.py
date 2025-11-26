import sys, os, time
sys.path.insert(0, "/app/faster-propainter-main")
os.chdir("/app/faster-propainter-main")

from model.misc import get_device, gpu_is_available
print(f"gpu_is_available: {gpu_is_available()}")
print(f"get_device: {get_device()}")

# Use existing cropped frames
frames_dir = "/app/test/cropped_frames"
masks_dir = "/app/test/cropped_masks"
output_dir = "/app/test/output_300f_gcc"

if not os.path.exists(frames_dir):
    print(f"[ERROR] {frames_dir} not found")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

frames_count = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
masks_count = len([f for f in os.listdir(masks_dir) if f.endswith(".png")])
print(f"Frames: {frames_count}, Masks: {masks_count}")

print(f"Starting ProPainter on {frames_count} frames...")
start = time.time()
from watermark import pipeline
pipeline(frames_dir, masks_dir, output_dir)
elapsed = time.time() - start
print(f"TOTAL: {elapsed:.2f}s for {frames_count} frames = {elapsed*1000/frames_count:.0f}ms/frame")
