"""
Test watermark removal with audio preservation
This uses the EXACT same code as server_production.py
"""

import sys
import os

# Add python_packages to path (same as server)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'python_packages'))

import cv2
import subprocess
import time

print("=" * 70)
print("WATERMARK REMOVAL WITH AUDIO TEST")
print("=" * 70)

# Paths
video_path = 'uploads/test_video.mp4'
output_temp = 'results/temp_no_audio.avi'
output_final = 'results/final_WITH_AUDIO.mp4'

if not os.path.exists(video_path):
    print(f"ERROR: {video_path} not found!")
    sys.exit(1)

# Load models
print("\n[1/4] Loading AI models...")

# Use simple OpenCV inpainting to avoid TensorRT issues
print("⚠️  Skipping AI models for this test - using simple inpainting")
print("   (This is just to test AUDIO preservation, not watermark quality)")

detector = None
inpainter = None

# Simple inpainting function
def simple_inpaint(frame, mask):
    """Simple OpenCV inpainting"""
    import cv2
    return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

print("✅ Using simple OpenCV inpainting")

# Process video frames
print("\n[2/4] Processing video frames (removing watermark)...")

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"   Video: {width}x{height} @ {fps}fps, {total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))

if not out.isOpened():
    print("ERROR: Failed to create video writer")
    sys.exit(1)

frames_processed = 0
frames_with_watermark = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frames_processed % 10 == 0:
        percent = int((frames_processed / total_frames) * 100)
        print(f'   Progress: {percent}% ({frames_processed}/{total_frames})', end='\r')

    # For this test, just write frames as-is (we're testing AUDIO, not watermark removal)
    # In real use, this would detect and remove watermarks
    out.write(frame)

    frames_processed += 1

cap.release()
out.release()

print(f'\n✅ Processed {frames_processed} frames')
print(f'   Output (no audio yet): {output_temp}')
print(f'   NOTE: This test copies video as-is to test AUDIO only')

# AUDIO MERGE - EXACT SAME CODE AS SERVER
print("\n[3/4] Merging audio from original (EXACT SERVER CODE)...")

try:
    # First check if original video has audio
    check_audio_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
    has_audio = 'audio' in has_audio_check.stdout

    if has_audio:
        print(f"✅ Original video has audio - merging...")
        cmd = [
            'ffmpeg',
            '-y',
            '-i', output_temp,
            '-i', video_path,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-strict', 'experimental',
            output_final
        ]
    else:
        print(f"⚠️  Original video has no audio")
        cmd = [
            'ffmpeg',
            '-y',
            '-i', output_temp,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            output_final
        ]

    print(f"   Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode == 0:
        print(f"✅ FFmpeg succeeded")

        # Verify output has audio
        verify_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            output_final
        ]
        verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)

        if 'audio' in verify.stdout:
            print(f"✅ VERIFIED: Output has audio!")
            # Delete temp file
            if os.path.exists(output_temp):
                os.remove(output_temp)
                print(f"   Cleaned up: {output_temp}")
        else:
            print(f"❌ WARNING: FFmpeg succeeded but output has NO audio!")
            print(f"   Stdout: {result.stdout}")
            print(f"   Stderr: {result.stderr}")
    else:
        print(f"❌ FFmpeg FAILED!")
        print(f"   Return code: {result.returncode}")
        print(f"   Stderr: {result.stderr}")
        print(f"   Stdout: {result.stdout}")

except FileNotFoundError as e:
    print(f"❌ FFmpeg not found: {e}")
except subprocess.TimeoutExpired:
    print(f"❌ FFmpeg timeout")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n[4/4] Final verification...")

# Check final output
if os.path.exists(output_final):
    # Get file size
    size_mb = os.path.getsize(output_final) / (1024 * 1024)
    print(f"✅ Output file exists: {output_final}")
    print(f"   Size: {size_mb:.2f} MB")

    # Check audio
    verify_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        output_final
    ]
    verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)

    if verify.stdout.strip():
        packets = verify.stdout.strip()
        print(f"✅ Audio track found: {packets} packets")
    else:
        print(f"❌ NO AUDIO in output!")

    print("\n" + "=" * 70)
    print("PLAY THIS FILE TO TEST:")
    print(f"  {output_final}")
    print("=" * 70)
else:
    print(f"❌ Output file not created!")

print("\nTest complete!")
