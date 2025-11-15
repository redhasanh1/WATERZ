"""
Process video with SAM2 masks using SEGMENT-BASED approach (from working commit 0ae8b014)
Each segment gets its own RAFT + ProPainter pass for better results
"""

import os
import sys
import cv2
import numpy as np
import shutil
import subprocess
from pathlib import Path
from tkinter import Tk, filedialog

# Add ProPainter
sys.path.insert(0, r'D:\watermarkz\faster-propainter-main')
from watermark import pipeline as faster_propainter_pipeline
from crop_utils import calculate_crop_region
from segment_detector import detect_segments_from_masks, merge_adjacent_segments

# Paths
MASKS_FOLDER = r"D:\watermarkz\temp_sam2_masks"
RESULT_DIR = r"D:\watermarkz\results"
TEMP_DIR = r"D:\watermarkz\temp"
FFMPEG_EXE = r"D:\watermarkz\ffmpeg\ffmpeg.exe"

def select_video():
    """Pick video with GUI"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    video_path = filedialog.askopenfilename(
        title="Select Video",
        initialdir=r"D:\watermarkz\videostotrain",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )

    root.destroy()
    return video_path if video_path else None

def main():
    print("="*80)
    print("SEGMENT-BASED LOCAL PROCESSING WITH MASKS")
    print("="*80)
    print()

    # Pick video
    video_path = select_video()
    if not video_path:
        print("No video selected!")
        return

    print(f"Video: {video_path}")

    # Check masks
    if not os.path.exists(MASKS_FOLDER):
        print(f"ERROR: No masks in {MASKS_FOLDER}")
        print("Run RUN_SAM2_WSL2_GUI.ps1 first!")
        return

    mask_files = sorted([f for f in os.listdir(MASKS_FOLDER) if f.endswith('.png')])
    print(f"Masks: {len(mask_files)} files")
    print()

    # Extract frames
    temp_prefix = f"seg_proc_{Path(video_path).stem}"
    frames_dir = Path(TEMP_DIR) / f"{temp_prefix}_frames"
    cropped_dir = Path(TEMP_DIR) / f"{temp_prefix}_cropped"
    sam2_masks_dir = Path(TEMP_DIR) / f"{temp_prefix}_masks"
    output_dir = Path(TEMP_DIR) / f"{temp_prefix}_output"

    for path in [frames_dir, cropped_dir, sam2_masks_dir, output_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print("[1/8] Extracting frames...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{frame_idx:04d}.png"), frame)
        frame_idx += 1
    cap.release()

    extracted_frames = frame_idx
    print(f"  {extracted_frames} frames @ {fps}fps ({width}x{height})")

    # Copy masks
    print(f"\n[2/8] Loading {len(mask_files)} masks...")
    for i, mask_file in enumerate(mask_files):
        src = Path(MASKS_FOLDER) / mask_file
        dst = sam2_masks_dir / f"{i:04d}.png"
        shutil.copy2(src, dst)

    # Calculate global crop from ALL masks
    print(f"\n[3/8] Analyzing all masks for crop region...")
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

    print(f"  Frames with masks: {masks_with_content}/{len(mask_files)}")

    if masks_with_content == 0:
        bbox = [0, 0, width, height]
    else:
        bbox = [min_x, min_y, max_x, max_y]

    # Calculate crop
    crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
    print(f"  Crop: {crop_x},{crop_y} {crop_w}x{crop_h}")

    # Crop frames and masks
    print(f"\n[4/8] Cropping frames and masks...")
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

    # Detect segments
    print(f"\n[5/8] Detecting segments from cropped masks...")
    segments = detect_segments_from_masks(
        str(sam2_masks_dir),
        position_tolerance=50,
        min_segment_length=3,
        max_segments=80
    )

    if len(segments) > 1:
        segments = merge_adjacent_segments(segments, position_tolerance=50, max_gap=60)

    print(f"  Found {len(segments)} segments")

    # Process segments
    print(f"\n[6/8] Processing {len(segments)} segments with ProPainter...")

    import torch
    use_fp16 = torch.cuda.is_available()

    if len(segments) <= 1:
        # Single segment - process entire video
        print(f"  Processing as single segment...")
        faster_propainter_pipeline(
            video=str(cropped_dir),
            mask=str(sam2_masks_dir),
            output=str(output_dir),
            resize_ratio=1.0,
            mask_dilation=4,
            ref_stride=15,
            neighbor_length=2,
            subvideo_length=40,
            raft_iter=10,
            mode="video_inpainting",
            save_frames=True,
            fp16=use_fp16
        )
        propainter_output = output_dir / cropped_dir.name / "frames"

    else:
        # Multiple segments - process each individually
        print(f"  Processing {len(segments)} segments individually...")

        for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
            duration = end_f - start_f
            movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
            is_stationary = movement < 10

            if is_stationary:
                neighbor_length, subvideo_length = 2, 20
                opt_label = "ULTRA-FAST"
            else:
                neighbor_length, subvideo_length = 2, 40
                opt_label = "BALANCED"

            print(f"\n  Segment {seg_idx+1}/{len(segments)}: frames {start_f}-{end_f} ({duration}f)")
            print(f"    Movement: {movement}px, {opt_label} (n={neighbor_length}, s={subvideo_length})")

            # Calculate segment-specific crop
            seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                seg_bbox, crop_w, crop_h, padding_ratio=0.15, min_size=128
            )

            # Extract segment frames/masks
            seg_frames_dir = output_dir / f"segment_{seg_idx}_frames"
            seg_masks_dir = output_dir / f"segment_{seg_idx}_masks"

            if seg_frames_dir.exists():
                shutil.rmtree(seg_frames_dir)
            if seg_masks_dir.exists():
                shutil.rmtree(seg_masks_dir)
            seg_frames_dir.mkdir()
            seg_masks_dir.mkdir()

            for frame_idx in range(start_f, end_f):
                frame_file = f"{frame_idx:04d}.png"
                src_frame = cropped_dir / frame_file
                src_mask = sam2_masks_dir / frame_file

                if src_frame.exists():
                    frame = cv2.imread(str(src_frame))
                    seg_frame = frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                    dst_frame = seg_frames_dir / f"{frame_idx-start_f:04d}.png"
                    cv2.imwrite(str(dst_frame), seg_frame)

                if src_mask.exists():
                    mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
                    seg_mask = mask[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                else:
                    seg_mask = np.zeros((seg_crop_h, seg_crop_w), dtype=np.uint8)

                dst_mask = seg_masks_dir / f"{frame_idx-start_f:04d}.png"
                cv2.imwrite(str(dst_mask), seg_mask)

            # Process this segment
            seg_output = output_dir / f"segment_{seg_idx}"
            seg_output.mkdir(exist_ok=True)

            faster_propainter_pipeline(
                video=str(seg_frames_dir),
                mask=str(seg_masks_dir),
                output=str(seg_output),
                resize_ratio=1.0,
                mask_dilation=4,
                ref_stride=15,
                neighbor_length=neighbor_length,
                subvideo_length=subvideo_length,
                raft_iter=10,
                mode="video_inpainting",
                save_frames=True,
                fp16=use_fp16
            )

        # Merge segments back
        print(f"\n[7/8] Merging {len(segments)} segments...")
        final_cropped_dir = output_dir / "final_cropped"
        final_cropped_dir.mkdir(exist_ok=True)

        for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
            seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                seg_bbox, crop_w, crop_h, padding_ratio=0.15, min_size=128
            )

            seg_output_frames = output_dir / f"segment_{seg_idx}" / f"segment_{seg_idx}_frames" / "frames"

            for frame_idx in range(start_f, end_f):
                result_frame_file = f"{frame_idx-start_f:04d}.png"
                result_path = seg_output_frames / result_frame_file

                # Load original cropped frame
                orig_cropped_path = cropped_dir / f"{frame_idx:04d}.png"
                orig_cropped = cv2.imread(str(orig_cropped_path))

                if result_path.exists() and orig_cropped is not None:
                    # Paste segment result back into cropped frame
                    seg_result = cv2.imread(str(result_path))
                    if seg_result is not None:
                        orig_cropped[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w] = seg_result

                    cv2.imwrite(str(final_cropped_dir / f"{frame_idx:04d}.png"), orig_cropped)

        propainter_output = final_cropped_dir

    # Merge back to full resolution
    print(f"\n[7/8] Merging to full resolution...")
    final_frames_dir = output_dir / "final_full"
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
    print(f"\n[8/8] Encoding final video...")
    output_path = Path(RESULT_DIR) / f"{Path(video_path).stem}_cleaned.mp4"
    output_path.parent.mkdir(exist_ok=True)

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
        '-c:a', 'copy',
        '-pix_fmt', 'yuv420p',
        '-y',
        str(output_path)
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    print()
    print("="*80)
    print("✅ DONE!")
    print("="*80)
    print(f"Output: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
    input("\nPress ENTER to exit...")
