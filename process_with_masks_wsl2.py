"""
Process video with existing SAM2 masks in WSL2
Uses all WSL2 optimizations (NeuFlow, TensorRT, etc.)

Usage:
    python process_with_masks_wsl2.py <video_path> <masks_folder>
"""

import os
import sys
import cv2
import numpy as np
import shutil
import subprocess
import json
from pathlib import Path

# Add ProPainter to path
BASE_DIR = Path(__file__).parent
PROPAINTER_DIR = BASE_DIR / "faster-propainter-main"
sys.path.insert(0, str(PROPAINTER_DIR))

from watermark import pipeline as faster_propainter_pipeline
from crop_utils import calculate_crop_region
from segment_detector import detect_segments, merge_adjacent_segments

# Paths
TEMP_DIR = BASE_DIR / "temp"
RESULT_DIR = BASE_DIR / "results"

# FFmpeg
FFMPEG_EXE = "ffmpeg"  # WSL2 uses system ffmpeg
FFPROBE_EXE = "ffprobe"


def get_video_metadata(video_path):
    """Get video metadata using FFprobe"""
    try:
        result = subprocess.run([
            FFPROBE_EXE,
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-of', 'json',
            video_path
        ], capture_output=True, text=True, timeout=10)

        data = json.loads(result.stdout)
        stream = data['streams'][0]

        width = int(stream['width'])
        height = int(stream['height'])

        # Parse frame rate
        fps_parts = stream['r_frame_rate'].split('/')
        fps = float(fps_parts[0]) / float(fps_parts[1])

        # Get frame count
        total_frames = int(stream.get('nb_frames', 0))

        return width, height, fps, total_frames
    except Exception as e:
        print(f"[WARNING] FFprobe failed: {e}")
        return None, None, None, None


def process_with_masks(video_path, masks_folder):
    """Process video with existing SAM2 masks"""

    print("\n" + "="*80)
    print("WSL2 PROCESSING WITH EXISTING SAM2 MASKS")
    print("="*80)

    # Check masks
    if not os.path.exists(masks_folder):
        print(f"\n[ERROR] Masks folder not found: {masks_folder}")
        return None

    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.png')])
    if len(mask_files) == 0:
        print(f"\n[ERROR] No mask files found in {masks_folder}")
        return None

    print(f"\n[OK] Video: {video_path}")
    print(f"[OK] Masks: {len(mask_files)} files")

    # Get video metadata
    width, height, fps, total_frames = get_video_metadata(video_path)

    if width is None:
        total_frames = len(mask_files)
        width, height, fps = 1920, 1080, 30.0

    print(f"[OK] Resolution: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Generate video ID
    video_id = Path(video_path).stem[:8]

    # Create temp directories
    temp_prefix = f"{video_id}_masks_wsl2"
    frames_dir = TEMP_DIR / f"{temp_prefix}_frames"
    cropped_dir = TEMP_DIR / f"{temp_prefix}_cropped"
    sam2_masks_dir = TEMP_DIR / f"{temp_prefix}_masks"
    output_dir = TEMP_DIR / f"{temp_prefix}_output"

    for path in [frames_dir, cropped_dir, sam2_masks_dir, output_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/7] Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video")
        return None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{frame_idx:04d}.png"), frame)
        frame_idx += 1
    cap.release()

    extracted_frames = frame_idx
    print(f"[OK] {extracted_frames} frames")

    print(f"\n[2/7] Loading masks...")
    for i, mask_file in enumerate(mask_files):
        src = Path(masks_folder) / mask_file
        dst = sam2_masks_dir / f"{i:04d}.png"
        shutil.copy2(src, dst)

    print(f"\n[3/7] Analyzing watermark region...")
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    masks_with_content = 0

    for i in range(len(mask_files)):
        mask_path = sam2_masks_dir / f"{i:04d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            white_pixels = np.sum(mask > 127)
            if white_pixels > 0:
                masks_with_content += 1
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)

    print(f"[OK] {masks_with_content}/{len(mask_files)} frames with masks")

    if masks_with_content == 0:
        bbox = [0, 0, width, height]
    else:
        bbox = [min_x, min_y, max_x, max_y]

    # Calculate crop
    crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
    print(f"[OK] Crop: {crop_x},{crop_y} {crop_w}x{crop_h}")

    print(f"\n[4/7] Cropping...")
    for i in range(extracted_frames):
        frame_file = f"{i:04d}.png"
        frame = cv2.imread(str(frames_dir / frame_file))
        if frame is not None:
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            cv2.imwrite(str(cropped_dir / frame_file), cropped)

    for i in range(len(mask_files)):
        mask_file = f"{i:04d}.png"
        mask = cv2.imread(str(sam2_masks_dir / mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            cv2.imwrite(str(sam2_masks_dir / mask_file), cropped_mask)

    print(f"\n[5/7] Detecting segments...")

    # Build detections from masks
    detections_per_frame = []
    for i in range(extracted_frames):
        mask_path = sam2_masks_dir / f"{i:04d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    detections_per_frame.append((x, y, x+w, y+h))
                else:
                    detections_per_frame.append(None)
            else:
                detections_per_frame.append(None)
        else:
            detections_per_frame.append(None)

    segments = detect_segments(
        detections_per_frame,
        position_tolerance=50,
        min_segment_length=3
    )

    if len(segments) > 1:
        segments = merge_adjacent_segments(segments, position_tolerance=50, max_gap=60)

    print(f"[OK] {len(segments)} segments")

    print(f"\n[6/7] Running ProPainter (WSL2 optimized)...")

    try:
        import torch
        use_fp16 = torch.cuda.is_available()

        # Simple single-pass processing (no complex segment merging)
        neighbor_length = 6
        subvideo_length = 40

        print(f"[OK] Processing with neighbor={neighbor_length}, subvideo={subvideo_length}")

        faster_propainter_pipeline(
            video=str(cropped_dir),
            mask=str(sam2_masks_dir),
            output=str(output_dir),
            resize_ratio=1.0,
            mask_dilation=4,
            ref_stride=15,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
            raft_iter=10,
            mode="video_inpainting",
            save_frames=True,
            fp16=use_fp16,
            frames_array=None,
            masks_array=None
        )

        propainter_output = output_dir / cropped_dir.name / "frames"

    except Exception as e:
        print(f"\n[ERROR] ProPainter failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"\n[7/7] Merging and encoding...")

    final_frames_dir = TEMP_DIR / f"{temp_prefix}_final"
    final_frames_dir.mkdir(exist_ok=True)

    for i in range(extracted_frames):
        frame_file = f"{i:04d}.png"
        orig_frame = cv2.imread(str(frames_dir / frame_file))

        if orig_frame is not None:
            cleaned_path = propainter_output / frame_file
            if cleaned_path.exists():
                cleaned_crop = cv2.imread(str(cleaned_path))
                if cleaned_crop is not None:
                    orig_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop

            cv2.imwrite(str(final_frames_dir / frame_file), orig_frame)

    # Encode
    output_path = RESULT_DIR / f"{video_id}_masks_removed.mp4"
    RESULT_DIR.mkdir(exist_ok=True)

    ffmpeg_cmd = [
        FFMPEG_EXE,
        '-framerate', str(fps),
        '-i', str(final_frames_dir / '%04d.png'),
        '-i', video_path,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-y',
        str(output_path)
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg failed: {result.stderr}")
        return None

    # Cleanup
    print(f"\n[CLEANUP] Removing temp files...")
    for temp_path in [frames_dir, cropped_dir, sam2_masks_dir, output_dir, final_frames_dir]:
        if temp_path.exists():
            shutil.rmtree(temp_path)

    return output_path


def main():
    if len(sys.argv) < 3:
        print("Usage: python process_with_masks_wsl2.py <video_path> <masks_folder>")
        sys.exit(1)

    video_path = sys.argv[1]
    masks_folder = sys.argv[2]

    output_path = process_with_masks(video_path, masks_folder)

    if output_path:
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print(f"Output: {output_path}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ FAILED")
        print("="*80)


if __name__ == "__main__":
    main()
