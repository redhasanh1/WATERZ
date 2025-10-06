import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import time
import os
import subprocess
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_optimized import LamaInpainterOptimized
from tqdm import tqdm

print("=" * 60)
print("FULL VIDEO WATERMARK REMOVAL TEST")
print("TensorRT YOLO + CUDA + OPTIMIZED LAMA + AUDIO")
print("=" * 60)

# Use test video
input_video = 'uploads/test_video.mp4'
temp_video_no_audio = 'temp_no_audio.mp4'  # Using mp4v codec for better compatibility
output_video = 'OUTPUT_WITH_AUDIO.mp4'

if not os.path.exists(input_video):
    print(f"❌ ERROR: {input_video} not found!")
    print("   Run SETUP_TEST.bat first to add a test video")
    exit()

print(f"\n📹 Input: {input_video}")
print(f"📁 Output: {output_video}")

# Initialize models
print("\n" + "=" * 60)
print("INITIALIZING")
print("=" * 60)
print("\nLoading TensorRT YOLO detector...")
detector = YOLOWatermarkDetector()

print("\nLoading OPTIMIZED LAMA inpainter...")
inpainter = LamaInpainterOptimized()

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

# Setup video writer (temporary AVI without audio)
# Use mp4v codec instead of XVID (more compatible)
temp_video_no_audio = 'temp_no_audio.mp4'  # Change to .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(temp_video_no_audio, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ Failed to create video writer!")
    exit()

print("\n" + "=" * 60)
print("PROCESSING VIDEO")
print("=" * 60)
print(f"\nUsing:")
print(f"  - TensorRT YOLO for detection (GPU accelerated)")
print(f"  - OPTIMIZED LAMA AI for inpainting (FP16 + CUDA)")
print(f"  - 0px padding (tight detection)")
print(f"  - Expected speedup: 1.5-2x faster than regular LAMA")
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

        # Inpaint
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

# ============================================================================
# AUDIO MERGE - SAME AS SERVER
# ============================================================================
print("\n" + "=" * 60)
print("MERGING AUDIO FROM ORIGINAL")
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

    print(f"\nRunning FFmpeg...")
    print(f"Command: {' '.join(cmd)}")
    print(f"\nThis may take a moment...")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    print(f"\nFFmpeg return code: {result.returncode}")

    if result.returncode == 0:
        print(f"✅ FFmpeg succeeded")

        # Verify output has audio
        verify_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            output_video
        ]
        verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)

        if 'audio' in verify.stdout:
            print(f"✅ VERIFIED: Output has AUDIO!")
            # Delete temp file
            if os.path.exists(temp_video_no_audio):
                os.remove(temp_video_no_audio)
                print(f"   Cleaned up temporary file")
        else:
            print(f"❌ WARNING: Output has NO audio!")
            print(f"   FFmpeg stdout: {result.stdout}")
            print(f"   FFmpeg stderr: {result.stderr}")
    else:
        print(f"\n❌ FFmpeg FAILED!")
        print(f"   Return code: {result.returncode}")
        print(f"\n   STDERR OUTPUT:")
        print(f"   {result.stderr}")
        print(f"\n   STDOUT OUTPUT:")
        print(f"   {result.stdout}")
        print(f"\n   Video saved WITHOUT audio: {temp_video_no_audio}")

except FileNotFoundError as e:
    print(f"❌ FFmpeg not found: {e}")
    print(f"   Video saved without audio: {temp_video_no_audio}")
except subprocess.TimeoutExpired:
    print(f"❌ FFmpeg timeout")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Final results
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

if os.path.exists(output_video):
    size_mb = os.path.getsize(output_video) / (1024 * 1024)
    print(f"\n✅ Output file: {output_video}")
    print(f"   Size: {size_mb:.2f} MB")

    # Check audio
    verify_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        output_video
    ]
    verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)

    if verify.stdout.strip():
        packets = verify.stdout.strip()
        print(f"   Audio: ✅ YES ({packets} packets)")
    else:
        print(f"   Audio: ❌ NO")

    print(f"\n💡 Play this video to verify:")
    print(f"   - Watermark is removed")
    print(f"   - Audio is preserved")
    print("=" * 60)
else:
    print(f"\n❌ Output file not created!")
    if os.path.exists(temp_video_no_audio):
        print(f"   Temp file (no audio): {temp_video_no_audio}")

print("\nTest complete!")
