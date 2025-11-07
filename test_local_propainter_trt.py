"""
Local TensorRT ProPainter Test - Replicates Celery workflow
Processes videostotrain/first.mp4 with TensorRT NeuFlow optimization
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
os.environ['YOLO_REQUIRE_TENSORRT'] = '1'

sys.path.insert(0, 'python_packages')
sys.path.insert(0, 'faster-propainter-main')

import cv2
import numpy as np
import time
import subprocess
from yolo_detector import YOLOWatermarkDetector
from watermark import pipeline as faster_propainter_pipeline
from tqdm import tqdm

print("=" * 80)
print("LOCAL TENSORRT PROPAINTER TEST")
print("TensorRT NeuFlow FP16 + YOLO Detection + ProPainter Inpainting")
print("=" * 80)

# Configuration
input_video = r'videostotrain\first.mp4'
output_dir = 'results/local_test'
final_output = 'results/first_no_watermark.mp4'

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs('results', exist_ok=True)

if not os.path.exists(input_video):
    print(f"[ERROR] {input_video} not found!")
    exit(1)

print(f"\nInput: {input_video}")
print(f"Output: {final_output}")

# ============================================================================
# PHASE 1: YOLO DETECTION (Batch processing like Celery)
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: YOLO BATCH DETECTION")
print("=" * 80)

print("\nInitializing TensorRT YOLO detector...")
start_init = time.time()
detector = YOLOWatermarkDetector()
init_time = time.time() - start_init
print(f"[OK] YOLO loaded in {init_time:.2f}s")

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

# Load all frames into memory (like Celery prepare_video_task)
print(f"\nLoading all frames into memory...")
all_frames = []
all_masks = []

start_load = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    all_frames.append(frame)
    frame_count += 1

cap.release()
load_time = time.time() - start_load
print(f"[OK] Loaded {frame_count} frames in {load_time:.2f}s")

# Batch YOLO detection (748 fps on RTX 4090)
print(f"\nRunning batch YOLO detection (batch_size=64)...")
start_detect = time.time()

# Process frames in batches
batch_size = 64
frames_with_watermark = 0

for i in tqdm(range(0, len(all_frames), batch_size), desc="Detecting watermarks"):
    batch_frames = all_frames[i:i+batch_size]

    # Batch detection
    batch_detections = detector.detect_batch(batch_frames, confidence_threshold=0.25, padding=0)

    # Create masks
    for frame, detections in zip(batch_frames, batch_detections):
        if detections:
            mask = detector.create_mask(frame, detections)
            frames_with_watermark += 1
        else:
            # No watermark - create empty mask
            mask = np.zeros((height, width), dtype=np.uint8)
        all_masks.append(mask)

detect_time = time.time() - start_detect
print(f"[OK] Detection complete in {detect_time:.2f}s ({detect_time/len(all_frames)*1000:.1f}ms per frame)")
print(f"   Frames with watermark: {frames_with_watermark}/{len(all_frames)}")

# ============================================================================
# PHASE 2: PROPAINTER INPAINTING (TensorRT NeuFlow)
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

# Run ProPainter with in-memory arrays (like Celery process_segment_task)
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
# PHASE 3: VIDEO FINALIZATION (Audio merge)
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
    print(f"[ERROR] ERROR: ProPainter output not found in {output_dir}")
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
        print(f"[WARNING]  Original video has no audio - copying video only...")
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
total_time = time.time() - start_init

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

if os.path.exists(final_output):
    size_mb = os.path.getsize(final_output) / (1024 * 1024)
    print(f"\n[OK] Final output: {final_output}")
    print(f"   Size: {size_mb:.2f} MB")

    # Check audio
    verify_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        final_output
    ]
    verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)

    if verify.stdout.strip():
        packets = verify.stdout.strip()
        print(f"   Audio: [OK] YES ({packets} packets)")
    else:
        print(f"   Audio: [ERROR] NO")

print(f"\nPerformance Summary:")
print(f"  - YOLO initialization: {init_time:.2f}s")
print(f"  - Frame loading: {load_time:.2f}s")
print(f"  - YOLO detection: {detect_time:.2f}s ({detect_time/len(all_frames)*1000:.1f}ms/frame)")
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
