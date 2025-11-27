import os
import cv2
from pathlib import Path

output_dir = Path('D:/watermarkz/output')

# Find latest video segments folder
segment_dirs = list(output_dir.glob('*_segments'))
if not segment_dirs:
    print('No segment folders found!')
    exit(1)

latest = max(segment_dirs, key=lambda x: x.stat().st_mtime)
video_id = latest.name.replace('_segments', '')
print(f'Assembling video: {video_id}')

# Get all segment folders sorted
segments = sorted(latest.glob('segment_*'), key=lambda x: int(x.name.split('_')[1]))
print(f'Found {len(segments)} segments')

# Collect all frames in order
all_frames = []
for seg in segments:
    frames = sorted(seg.glob('*.png'), key=lambda x: int(x.stem))
    all_frames.extend(frames)
    print(f'  {seg.name}: {len(frames)} frames')

print(f'Total frames: {len(all_frames)}')

if not all_frames:
    print('No frames found!')
    exit(1)

# Read first frame to get dimensions
first_frame = cv2.imread(str(all_frames[0]))
h, w = first_frame.shape[:2]
print(f'Frame size: {w}x{h}')

# Create video writer
output_file = output_dir / f'{video_id}_final.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_file), fourcc, 30.0, (w, h))

print('Writing video...')
for i, frame_path in enumerate(all_frames):
    frame = cv2.imread(str(frame_path))
    out.write(frame)
    if (i + 1) % 50 == 0:
        print(f'  {i + 1}/{len(all_frames)} frames written')

out.release()
print(f'Done! Output: {output_file}')
