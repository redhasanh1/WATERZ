import sys
sys.path.insert(0, 'python_packages')

import cv2
import os
import glob
import json

print("=" * 60)
print("Extract Frames from Sora Videos")
print("=" * 60)

# Paths
videos_dir = 'videostotrain'  # Videos to train folder
output_dir = 'NEW_SORA_TRAINING/images'
progress_file = 'NEW_SORA_TRAINING/extracted_videos.json'

os.makedirs(videos_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(progress_file), exist_ok=True)

# Load progress (list of already extracted videos)
extracted_videos = []
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        extracted_videos = json.load(f)
    print(f"📋 Loaded progress: {len(extracted_videos)} video(s) already extracted")

# Find all videos
video_files = glob.glob(os.path.join(videos_dir, '*.mp4'))

if not video_files:
    print(f"❌ No videos found in {videos_dir}/")
    print(f"Put Sora videos in the {videos_dir}/ folder first!")
    exit()

print(f"\nFound {len(video_files)} video(s) in {videos_dir}/")
print(f"Output directory: {output_dir}/")
print(f"Extracting exactly 30 frames per video")
print()

# Process each video
total_frames_saved = 0
videos_processed = 0

for video_idx, video_path in enumerate(video_files):
    video_name = os.path.basename(video_path)

    # Skip if already extracted
    if video_name in extracted_videos:
        print(f"[{video_idx + 1}/{len(video_files)}] {video_name} - ✅ Already extracted, skipping...")
        continue

    print(f"\n[{video_idx + 1}/{len(video_files)}] Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Failed to open video")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  📊 FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")

    # Calculate interval to extract exactly 30 frames
    target_frames = 30
    if total_frames < target_frames:
        print(f"  ⚠️  Video has less than {target_frames} frames, extracting all frames")
        interval = 1
    else:
        interval = total_frames // target_frames

    print(f"  📸 Extracting every {interval} frame(s) to get ~{target_frames} frames")

    frame_count = 0
    saved_count = 0

    # Count existing frames to continue numbering
    existing_frames = len(glob.glob(os.path.join(output_dir, 'sora_frame_*.jpg')))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frames at calculated interval
        if frame_count % interval == 0 and saved_count < target_frames:
            output_path = os.path.join(output_dir, f'sora_frame_{existing_frames + saved_count:04d}.jpg')
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"  ✅ Extracted {saved_count} frames")
    total_frames_saved += saved_count
    videos_processed += 1

    # Update progress
    extracted_videos.append(video_name)
    with open(progress_file, 'w') as f:
        json.dump(extracted_videos, f, indent=2)

print(f"\n✅ Processed {videos_processed} new video(s)")
print(f"✅ Total frames extracted this run: {total_frames_saved}")
print(f"📊 Progress saved to: {progress_file}")
print(f"🖼️  All frames saved to: {output_dir}/")
print(f"\nNext step: Run LABEL_SORA.bat to label the watermarks!")
