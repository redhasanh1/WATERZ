"""
Simple test to check if audio is being preserved
Run this to debug audio issues
"""
import subprocess
import os
import sys

print("=" * 60)
print("Audio Preservation Test")
print("=" * 60)

# Check FFmpeg
print("\n1. Checking FFmpeg installation...")
try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ FFmpeg is installed")
        print(f"   Version: {result.stdout.split('\\n')[0]}")
    else:
        print("❌ FFmpeg not working properly")
        sys.exit(1)
except FileNotFoundError:
    print("❌ FFmpeg NOT found!")
    print("\nPlease install FFmpeg:")
    print("  Windows: https://www.gyan.dev/ffmpeg/builds/")
    print("  Download ffmpeg-release-essentials.zip")
    print("  Extract and add to PATH")
    sys.exit(1)

# Check if test video exists
print("\n2. Checking for test video...")
test_video = "uploads/test_video.mp4"

if not os.path.exists(test_video):
    print(f"❌ Test video not found: {test_video}")
    print("\nPlease place a video file at: uploads/test_video.mp4")
    sys.exit(1)

print(f"✅ Test video found: {test_video}")

# Check if video has audio
print("\n3. Checking if test video has audio...")
cmd = [
    'ffprobe',
    '-v', 'error',
    '-select_streams', 'a:0',
    '-show_entries', 'stream=codec_type',
    '-of', 'default=noprint_wrappers=1:nokey=1',
    test_video
]

result = subprocess.run(cmd, capture_output=True, text=True)
if 'audio' in result.stdout:
    print("✅ Test video HAS audio")
else:
    print("⚠️  Test video has NO audio!")
    print("   Please use a video with audio for testing")

# Test audio extraction and merge
print("\n4. Testing audio extraction...")

os.makedirs('results', exist_ok=True)

# Extract audio only
audio_file = 'results/test_audio.aac'
cmd = [
    'ffmpeg', '-y',
    '-i', test_video,
    '-vn',  # No video
    '-acodec', 'copy',
    audio_file
]

print("   Extracting audio...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0 and os.path.exists(audio_file):
    print(f"✅ Audio extracted: {audio_file}")
else:
    print(f"❌ Audio extraction failed")
    print(f"   Error: {result.stderr}")

# Create a copy with audio merged (simple test)
print("\n5. Testing audio merge...")
output_test = 'results/test_merge.mp4'

cmd = [
    'ffmpeg', '-y',
    '-i', test_video,  # Video input
    '-i', test_video,  # Audio input (same file for test)
    '-map', '0:v',     # Video from first input
    '-map', '1:a',     # Audio from second input
    '-c:v', 'copy',    # Copy video codec (no re-encoding)
    '-c:a', 'aac',     # Encode audio to AAC
    '-b:a', '192k',
    output_test
]

print("   Merging audio...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"✅ Audio merge successful: {output_test}")
    print("\n6. Verifying output has audio...")

    # Check output audio
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        output_test
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if 'audio' in result.stdout:
        print("✅ Output video HAS audio!")
        print("\n" + "=" * 60)
        print("SUCCESS! Audio pipeline is working")
        print("=" * 60)
        print(f"\nPlay this file to verify: {output_test}")
    else:
        print("❌ Output video has NO audio")
else:
    print(f"❌ Audio merge failed")
    print(f"   Error: {result.stderr}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
