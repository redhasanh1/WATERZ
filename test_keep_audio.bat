@echo off
echo ================================================================
echo WATERMARK REMOVAL - AUDIO PRESERVATION TEST
echo ================================================================
echo.
echo This script will:
echo   1. Remove watermark from video
echo   2. KEEP the original audio (not remove it!)
echo   3. Output video with audio intact
echo.
echo ================================================================
pause

REM Check FFmpeg
echo.
echo Checking FFmpeg...
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] FFmpeg not installed!
    echo.
    echo Download from: https://www.gyan.dev/ffmpeg/builds/
    echo Install and add to PATH
    pause
    exit /b 1
)
echo [OK] FFmpeg found

REM Check test video
echo.
echo Checking for test video...
if not exist uploads\test_video.mp4 (
    echo [ERROR] Test video not found!
    echo Please place a video with audio at: uploads\test_video.mp4
    pause
    exit /b 1
)
echo [OK] Test video found

REM Check if video has audio
echo.
echo Checking if video HAS audio...
ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 uploads\test_video.mp4 > temp_check.txt 2>&1
findstr /C:"audio" temp_check.txt >nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Test video has NO audio!
    echo Please use a video that has audio for testing
    del temp_check.txt
    pause
    exit /b 1
)
del temp_check.txt
echo [OK] Test video HAS audio - we will KEEP it!

echo.
echo ================================================================
echo Starting watermark removal (KEEPING AUDIO)...
echo ================================================================
echo.

REM Create folders
if not exist uploads mkdir uploads
if not exist results mkdir results

REM Activate venv if exists
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

REM Run processing
python -c "
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))

print('=' * 70)
print('STEP 1: Loading AI models...')
print('=' * 70)
from yolo_detector import YOLOWatermarkDetector
try:
    from lama_inpaint_optimized import LamaInpainterOptimized as LamaInpainter
    print('Using optimized LaMa')
except:
    from lama_inpaint_local import LamaInpainter
    print('Using standard LaMa')

import cv2
import subprocess

detector = YOLOWatermarkDetector()
inpainter = LamaInpainter()

print('')
print('=' * 70)
print('STEP 2: Processing video frames (removing watermark)...')
print('=' * 70)

video_path = 'uploads/test_video.mp4'
temp_no_audio = 'results/temp_no_audio.avi'
final_with_audio = 'results/output_WITH_AUDIO.mp4'

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'Video: {width}x{height} @ {fps}fps, {total_frames} frames')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_no_audio, fourcc, fps, (width, height))

frame_count = 0
watermark_count = 0

print('Processing frames...')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count %% 10 == 0:
        percent = int((frame_count / total_frames) * 100)
        print(f'  Progress: {percent}%% ({frame_count}/{total_frames} frames)', end='\r')

    detections = detector.detect(frame, confidence_threshold=0.25, padding=0)

    if detections:
        watermark_count += 1
        mask = detector.create_mask(frame, detections)
        processed_frame = inpainter.inpaint_region(frame, mask)
        out.write(processed_frame)
    else:
        out.write(frame)

    frame_count += 1

cap.release()
out.release()

print('')
print(f'Processed {frame_count} frames')
print(f'Found watermarks in {watermark_count} frames')
print('')
print('NOTE: At this point, video has NO audio (OpenCV limitation)')

print('')
print('=' * 70)
print('STEP 3: MERGING AUDIO from original video...')
print('=' * 70)
print('This is the IMPORTANT step - we KEEP the audio!')

# Merge audio from ORIGINAL video
cmd = [
    'ffmpeg', '-y',
    '-i', temp_no_audio,  # Processed video (no watermark, no audio)
    '-i', video_path,     # Original video (with audio)
    '-map', '0:v:0',      # Take VIDEO from processed (watermark removed)
    '-map', '1:a:0',      # Take AUDIO from original (KEEP IT!)
    '-c:v', 'libx264',    # Encode video
    '-preset', 'ultrafast',
    '-crf', '18',
    '-c:a', 'aac',        # Encode audio
    '-b:a', '192k',
    '-strict', 'experimental',
    final_with_audio
]

print('Running FFmpeg to merge audio...')
print(f'Command: {\" \".join(cmd)}')
print('')

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print('✅ SUCCESS! Audio has been KEPT in the output video!')
    print('')

    # Clean up temp file
    if os.path.exists(temp_no_audio):
        os.remove(temp_no_audio)
        print('Cleaned up temporary file (without audio)')

    print('')
    print('=' * 70)
    print('VERIFICATION: Checking if output has audio...')
    print('=' * 70)

    # Verify output has audio
    verify_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        final_with_audio
    ]

    verify = subprocess.run(verify_cmd, capture_output=True, text=True)
    if 'audio' in verify.stdout:
        print('✅✅✅ CONFIRMED: Output video HAS AUDIO! ✅✅✅')
    else:
        print('❌ WARNING: Output video has no audio')

    print('')
    print('=' * 70)
    print('OUTPUT FILE: results\\output_WITH_AUDIO.mp4')
    print('=' * 70)
else:
    print('❌ FFmpeg failed!')
    print('Error:', result.stderr)
    print('')
    print('Output saved without audio: results\\temp_no_audio.avi')
"

echo.
echo ================================================================
echo TEST COMPLETE!
echo ================================================================
echo.
echo Result: results\output_WITH_AUDIO.mp4
echo.
echo PLAY THIS VIDEO to verify:
echo   - Watermark is REMOVED
echo   - Audio is KEPT (not removed!)
echo.
pause
