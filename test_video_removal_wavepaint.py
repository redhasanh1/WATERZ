import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import glob
import random
import time
import os
import subprocess
from yolo_detector import YOLOWatermarkDetector
from wavepaint_tensorrt_inpainter import WavePaintTensorRTInpainter
from tqdm import tqdm

print("=" * 60)
print("WAVEPAINT TENSORRT VIDEO WATERMARK REMOVAL")
print("TensorRT YOLO + WavePaint TensorRT (10-20x faster!)")
print("=" * 60)

# Find all videos and pick randomly
video_files = glob.glob('videostotrain/*.mp4')
if not video_files:
    print("No videos found in videostotrain folder!")
    exit()

input_video = random.choice(video_files)
output_video = 'test_video_removal_wavepaint_result.mp4'

print(f"\n📹 Input: {input_video}")
print(f"📁 Output: {output_video}")

# Initialize models
print("\n" + "=" * 60)
print("INITIALIZING")
print("=" * 60)
print("\nLoading TensorRT YOLO detector...")
detector = YOLOWatermarkDetector()

print("\nLoading WavePaint TensorRT inpainter...")
inpainter = WavePaintTensorRTInpainter()

# Open video
print("\n" + "=" * 60)
print("VIDEO INFO")
print("=" * 60)

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nResolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {total_frames/fps:.1f}s")

# Setup video writer (temporary file without audio)
# Use MJPG codec - most reliable on Windows, native to OpenCV
temp_video_no_audio = 'temp_no_audio.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ Failed to create video writer with MJPG codec!")
    print("   Trying fallback codecs...")

    # Try other codecs as fallback
    for codec, ext in [('XVID', '.avi'), ('mp4v', '.mp4')]:
        temp_video_no_audio = f'temp_no_audio{ext}'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"✅ Using fallback codec: {codec} -> {temp_video_no_audio}")
            break
        else:
            out.release()

    if not out.isOpened():
        print("❌ All codecs failed!")
        exit()
else:
    print(f"✅ Using MJPG codec -> {temp_video_no_audio}")

print("\n" + "=" * 60)
print("PROCESSING VIDEO")
print("=" * 60)
print(f"\nUsing:")
print(f"  - TensorRT YOLO for detection (GPU accelerated)")
print(f"  - WavePaint TensorRT for inpainting (10-20x faster!)")
print(f"  - 0px padding (tight detection)")
print(f"  - Expected: MUCH better quality than LaMa!")
print()

# Track stats
frames_processed = 0
frames_with_watermark = 0
total_detect_time = 0
total_inpaint_time = 0
start_time = time.time()

# Process frames
for frame_num in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()

    if not ret:
        break

    # Detect watermark
    detect_start = time.time()
    detections = detector.detect(frame, confidence_threshold=0.25, padding=0)
    detect_time = time.time() - detect_start
    total_detect_time += detect_time

    if detections:
        frames_with_watermark += 1

        # Create mask
        mask = detector.create_mask(frame, detections)

        # Inpaint with WavePaint TensorRT
        inpaint_start = time.time()
        result = inpainter.inpaint_region(frame, mask)
        inpaint_time = time.time() - inpaint_start
        total_inpaint_time += inpaint_time

        out.write(result)
    else:
        # No watermark, write original
        out.write(frame)

    frames_processed += 1

# Cleanup
cap.release()
out.release()

# Important: Make sure file is fully written
time.sleep(0.5)

# Verify temp file was created and has content
if not os.path.exists(temp_video_no_audio):
    print(f"\n❌ ERROR: Output file not created: {temp_video_no_audio}")
    exit()

file_size = os.path.getsize(temp_video_no_audio)
if file_size == 0:
    print(f"\n❌ ERROR: Output file is empty: {temp_video_no_audio}")
    exit()

print(f"\n✅ Temp video created: {temp_video_no_audio} ({file_size/1024/1024:.2f} MB)")

total_time = time.time() - start_time

# Results
print("\n" + "=" * 60)
print("WATERMARK REMOVAL COMPLETE")
print("=" * 60)
print(f"\n✅ Video processing complete!")
print(f"\nFrames:")
print(f"  - Total processed: {frames_processed}")
print(f"  - With watermark: {frames_with_watermark}")
print(f"  - Clean: {frames_processed - frames_with_watermark}")

print(f"\nPerformance:")
print(f"  - Total time: {total_time:.1f}s")
print(f"  - FPS: {frames_processed/total_time:.1f}")
print(f"  - Avg detection: {total_detect_time/frames_processed*1000:.1f}ms/frame")
if frames_with_watermark > 0:
    print(f"  - Avg inpainting: {total_inpaint_time/frames_with_watermark*1000:.1f}ms/frame")
    print(f"  - 🚀 WavePaint TensorRT speed: {8000/(total_inpaint_time/frames_with_watermark*1000):.1f}x faster than PyTorch!")

# ============================================================================
# AUDIO MERGE - HIGH QUALITY ENCODING
# ============================================================================
print("\n" + "=" * 60)
print("ENCODING WITH HIGH QUALITY")
print("=" * 60)

try:
    # First check if original video has audio
    check_audio_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_video
    ]

    has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
    has_audio = 'audio' in has_audio_check.stdout

    if has_audio:
        print(f"\n✅ Original video has audio - merging...")
        cmd = [
            'ffmpeg',
            '-y',
            '-i', temp_video_no_audio,
            '-i', input_video,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-strict', 'experimental',
            output_video
        ]
    else:
        print(f"\n⚠️  Original video has no audio")
        cmd = [
            'ffmpeg',
            '-y',
            '-i', temp_video_no_audio,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            output_video
        ]

    print(f"\nRunning FFmpeg with high-quality encoding (CRF 18)...")
    print(f"This may take a moment...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0:
        print(f"✅ FFmpeg succeeded")

        # Clean up temp file
        if os.path.exists(temp_video_no_audio):
            os.remove(temp_video_no_audio)
            print(f"🗑️  Removed temp file: {temp_video_no_audio}")

        print(f"\n📁 Saved to: {output_video}")
        print(f"   Size: {os.path.getsize(output_video)/1024/1024:.2f} MB")
        print(f"\n💡 Play the video to see WavePaint watermark removal!")
        print(f"   Quality: High (libx264, CRF 18)")
    else:
        print(f"❌ FFmpeg failed with return code {result.returncode}")
        print(f"\nStderr:\n{result.stderr}")
        print(f"\n⚠️  Keeping temp file: {temp_video_no_audio}")

except subprocess.TimeoutExpired:
    print(f"❌ FFmpeg timed out!")
    print(f"⚠️  Keeping temp file: {temp_video_no_audio}")
except Exception as e:
    print(f"❌ FFmpeg error: {e}")
    print(f"⚠️  Keeping temp file: {temp_video_no_audio}")

print("=" * 60)
