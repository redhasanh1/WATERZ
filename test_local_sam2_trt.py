"""
Local TensorRT ProPainter Test with SAM2 Masks
EXACT SAME as test_local_propainter_trt.py but uses SAM2 masks instead of YOLO
"""
import sys
import os

# Add TensorRT DLL path BEFORE any imports
trt_lib_path = r"D:\watermarkz\TensorRT-10.13.3.9\lib"
if trt_lib_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = trt_lib_path + os.pathsep + os.environ.get('PATH', '')

# Enable TensorRT NeuFlow (CRITICAL!)
os.environ['USE_NEUFLOW'] = '1'  # Use NeuFlow ONNX for optical flow (10-70x faster!)
os.environ['FORCE_TRT_RAFT'] = '0'
os.environ['ENABLE_FLASH_ATTENTION'] = '1'
os.environ['SEGMENT_WORKERS'] = '4'

sys.path.insert(0, 'python_packages')
sys.path.insert(0, 'faster-propainter-main')

import cv2
import numpy as np
import time
import subprocess
from watermark import pipeline as faster_propainter_pipeline
from tqdm import tqdm
from tkinter import Tk, filedialog

print("=" * 80)
print("LOCAL TENSORRT PROPAINTER TEST WITH SAM2 MASKS")
print("TensorRT NeuFlow FP16 + SAM2 Masks + ProPainter Inpainting")
print("=" * 80)

# Configuration
MASKS_DIR = r'D:\watermarkz\temp_sam2_masks_expanded'
output_dir = 'results/sam2_local_test'
final_output = 'results/sam2_cleaned.mp4'

# Pick video with GUI
def select_video():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    video_path = filedialog.askopenfilename(
        title="Select Video for SAM2 Inpainting",
        initialdir=r"D:\watermarkz\videostotrain",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    root.destroy()
    return video_path if video_path else None

input_video = select_video()
if not input_video:
    print("[ERROR] No video selected!")
    exit(1)

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs('results', exist_ok=True)

print(f"\nInput: {input_video}")
print(f"Masks: {MASKS_DIR}")
print(f"Output: {final_output}")

# ============================================================================
# PHASE 1: LOAD SAM2 MASKS (instead of YOLO detection)
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: LOAD SAM2 MASKS")
print("=" * 80)

if not os.path.exists(MASKS_DIR):
    print(f"[ERROR] SAM2 masks not found at {MASKS_DIR}")
    print("Run EXPAND_MASKS_TO_FILES.py first!")
    exit(1)

# Load video
print("\nOpening video...")
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo Info:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total frames: {total_frames}")
print(f"  Duration: {total_frames/fps:.1f}s")

# Load all frames into memory
print(f"\nLoading all frames into memory...")
all_frames = []
start_load = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    all_frames.append(frame)
cap.release()
load_time = time.time() - start_load
print(f"[OK] Loaded {len(all_frames)} frames in {load_time:.2f}s")

# Load SAM2 masks
print(f"\nLoading SAM2 masks from {MASKS_DIR}...")
mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith('.png')])
print(f"Found {len(mask_files)} mask files")

all_masks = []
start_masks = time.time()
for mask_file in tqdm(mask_files, desc="Loading masks"):
    mask = cv2.imread(os.path.join(MASKS_DIR, mask_file), cv2.IMREAD_GRAYSCALE)
    all_masks.append(mask)
mask_time = time.time() - start_masks
print(f"[OK] Loaded {len(all_masks)} masks in {mask_time:.2f}s")

# Verify mask count matches frame count
if len(all_masks) != len(all_frames):
    print(f"[WARNING] Mask count ({len(all_masks)}) != Frame count ({len(all_frames)})")
    # Truncate to shorter
    min_count = min(len(all_masks), len(all_frames))
    all_frames = all_frames[:min_count]
    all_masks = all_masks[:min_count]
    print(f"   Using first {min_count} frames/masks")

# Count non-empty masks
frames_with_mask = sum(1 for m in all_masks if np.sum(m > 127) > 0)
print(f"   Frames with watermark mask: {frames_with_mask}/{len(all_masks)}")

# ============================================================================
# PHASE 2: PROPAINTER INPAINTING (TensorRT NeuFlow) - EXACT SAME AS YOLO VERSION
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: PROPAINTER INPAINTING WITH TENSORRT NEUFLOW")
print("=" * 80)

print(f"\nConfiguration:")
print(f"  - Optical Flow: TensorRT NeuFlow FP16 (6.6ms per frame)")
print(f"  - Precision: FP16 (2x speedup)")
print(f"  - Mask dilation: 4px")
print(f"  - Reference stride: 15")
print(f"  - Neighbor length: 10")
print(f"  - Subvideo length: 120")
print(f"  - In-memory processing: YES (ZERO disk I/O)")
print()

start_propainter = time.time()

# Make arrays contiguous (required for cuDNN)
all_frames_contiguous = [np.ascontiguousarray(f) for f in all_frames]
all_masks_contiguous = [np.ascontiguousarray(m) for m in all_masks]

# Run ProPainter with in-memory arrays - EXACT SAME CALL AS YOLO VERSION
try:
    faster_propainter_pipeline(
        video=input_video,  # Used for metadata only
        mask='dummy_mask',  # Not used when masks_array provided
        output=output_dir,
        resize_ratio=1.0,
        mask_dilation=4,
        ref_stride=15,
        neighbor_length=10,
        subvideo_length=120,
        raft_iter=10,
        mode="video_inpainting",
        save_fps=fps,
        save_frames=True,
        fp16=True,  # FP16 for 2x speedup
        use_cached_models=True,  # Reuse loaded models
        frames_array=all_frames_contiguous,  # Direct memory input (skips disk I/O!)
        masks_array=all_masks_contiguous
    )

    propainter_time = time.time() - start_propainter
    print(f"\n[OK] ProPainter complete in {propainter_time:.2f}s ({propainter_time/len(all_frames)*1000:.1f}ms per frame)")

except Exception as e:
    print(f"\n[ERROR] ProPainter failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# PHASE 3: VIDEO FINALIZATION (Audio merge) - EXACT SAME AS YOLO VERSION
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3: FINALIZE VIDEO WITH AUDIO")
print("=" * 80)

# Find ProPainter output
propainter_output = None
for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file == 'inpaint_out.mp4':
            propainter_output = os.path.join(root, file)
            break
    if propainter_output:
        break

if not propainter_output or not os.path.exists(propainter_output):
    print(f"[ERROR] ProPainter output not found in {output_dir}")
    exit(1)

print(f"\n[OK] ProPainter output: {propainter_output}")
print(f"   Size: {os.path.getsize(propainter_output)/1024/1024:.2f} MB")

# Check if original has audio
print(f"\nChecking for audio in original video...")
check_audio_cmd = [
    'ffprobe',
    '-v', 'error',
    '-select_streams', 'a:0',
    '-show_entries', 'stream=codec_type',
    '-of', 'default=noprint_wrappers=1:nokey=1',
    input_video
]

try:
    has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
    has_audio = 'audio' in has_audio_check.stdout

    if has_audio:
        print(f"[OK] Original video has audio - merging...")
        cmd = [
            'ffmpeg',
            '-y',
            '-i', propainter_output,
            '-i', input_video,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-strict', 'experimental',
            final_output
        ]
    else:
        print(f"[WARNING] Original video has no audio - copying video only...")
        cmd = [
            'ffmpeg',
            '-y',
            '-i', propainter_output,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            final_output
        ]

    print(f"\nRunning FFmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0:
        print(f"[OK] FFmpeg succeeded")
    else:
        print(f"[ERROR] FFmpeg failed (return code: {result.returncode})")
        print(f"   STDERR: {result.stderr[:500]}")

except Exception as e:
    print(f"[ERROR] Audio merge failed: {e}")

# ============================================================================
# FINAL RESULTS
# ============================================================================
total_time = time.time() - start_load

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

if os.path.exists(final_output):
    size_mb = os.path.getsize(final_output) / (1024 * 1024)
    print(f"\n[OK] Final output: {final_output}")
    print(f"   Size: {size_mb:.2f} MB")

print(f"\nPerformance Summary:")
print(f"  - Frame loading: {load_time:.2f}s")
print(f"  - Mask loading: {mask_time:.2f}s")
print(f"  - ProPainter: {propainter_time:.2f}s ({propainter_time/len(all_frames)*1000:.1f}ms/frame)")
print(f"  - Total time: {total_time:.2f}s")
print(f"  - Overall FPS: {len(all_frames)/total_time:.1f}")

print("\n" + "=" * 80)
print("TEST COMPLETE!")
print("=" * 80)
print(f"\n[INFO] Play {final_output} to verify:")
print(f"   - Watermark is removed")
print(f"   - Audio is preserved")
print(f"   - Quality is maintained")
