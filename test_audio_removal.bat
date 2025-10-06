@echo off
echo ========================================
echo Watermark Removal Audio Test
echo ========================================
echo.
echo This will test video watermark removal while preserving audio
echo.

REM Check if FFmpeg is installed
where ffmpeg >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] FFmpeg not found!
    echo.
    echo Please install FFmpeg:
    echo 1. Download from: https://ffmpeg.org/download.html
    echo 2. Extract to C:\ffmpeg
    echo 3. Add C:\ffmpeg\bin to your PATH
    echo.
    pause
    exit /b 1
)

echo [OK] FFmpeg found
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Starting test...
echo.
echo Instructions:
echo 1. Place a test video file in the 'uploads' folder
echo 2. Name it 'test_video.mp4'
echo 3. The script will process it and save to 'results' folder
echo 4. Check the output video has audio!
echo.

REM Create folders if they don't exist
if not exist uploads mkdir uploads
if not exist results mkdir results

REM Check if test video exists
if not exist uploads\test_video.mp4 (
    echo [ERROR] Test video not found!
    echo Please place a video file at: uploads\test_video.mp4
    echo.
    pause
    exit /b 1
)

echo [OK] Test video found: uploads\test_video.mp4
echo.

REM Run Python test script
python -c "
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))

print('Loading AI models...')
from yolo_detector import YOLOWatermarkDetector
try:
    from lama_inpaint_optimized import LamaInpainterOptimized as LamaInpainter
    print('Using optimized LaMa')
except:
    from lama_inpaint_local import LamaInpainter
    print('Using standard LaMa')

import cv2
import subprocess
import time

detector = YOLOWatermarkDetector()
inpainter = LamaInpainter()

print('Processing video...')
video_path = 'uploads/test_video.mp4'
output_temp = 'results/test_output_temp.avi'
output_final = 'results/test_output_with_audio.mp4'

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f'Video: {width}x{height} @ {fps}fps, {total_frames} frames')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))

frame_count = 0
watermark_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 10 == 0:
        print(f'Frame {frame_count}/{total_frames}...', end='\r')

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

print(f'\nProcessed {frame_count} frames, found watermarks in {watermark_count} frames')
print('Merging audio...')

# Merge audio using FFmpeg
cmd = [
    'ffmpeg', '-y',
    '-i', output_temp,
    '-i', video_path,
    '-map', '0:v',
    '-map', '1:a?',
    '-c:v', 'libx264',
    '-preset', 'fast',
    '-crf', '23',
    '-c:a', 'aac',
    '-b:a', '192k',
    '-shortest',
    output_final
]

result = subprocess.run(cmd, capture_output=True)

if result.returncode == 0:
    print('✅ SUCCESS! Audio merged')
    os.remove(output_temp)
    print(f'\nOutput saved to: {output_final}')
    print('\nPlay the video to verify audio is present!')
else:
    print('❌ FFmpeg failed:', result.stderr.decode())
"

echo.
echo ========================================
echo Test complete!
echo ========================================
echo.
echo Output file: results\test_output_with_audio.mp4
echo.
echo Play the video to check if audio is preserved!
echo.
pause
