"""
FASTEST LOCAL PROCESSING - Uses NeuFlow TensorRT (10-70x faster than RAFT)
Simple single-pass processing with all optimizations enabled
"""

import os
import sys
import cv2
import numpy as np
import subprocess
import shutil
from pathlib import Path
from tkinter import Tk, filedialog

# Enable all speed optimizations
os.environ['USE_NEUFLOW'] = '1'  # 10-70x faster optical flow
os.environ['ENABLE_FP8_TRANSFORMER'] = '1'  # 1.3-1.5x speedup
os.environ['ENABLE_FLASH_ATTENTION'] = '1'  # 3-5x transformer speedup

# Add ProPainter
sys.path.insert(0, r'D:\watermarkz\faster-propainter-main')
from watermark import pipeline as faster_propainter_pipeline
from crop_utils import calculate_crop_region

# Paths
MASKS_FOLDER = r"D:\watermarkz\temp_sam2_masks"
RESULT_DIR = r"D:\watermarkz\results"
TEMP_DIR = r"D:\watermarkz\temp"

def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths with fallback to static-ffmpeg."""
    # Try system PATH first
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')

    if ffmpeg_path and ffprobe_path:
        print(f"[OK] Using system FFmpeg: {ffmpeg_path}")
        return ffmpeg_path, ffprobe_path

    # Fallback to static-ffmpeg
    try:
        from static_ffmpeg import run
        ffmpeg_path, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
        print(f"[OK] Using static-ffmpeg: {ffmpeg_path}")
        return ffmpeg_path, ffprobe_path
    except ImportError:
        raise RuntimeError("FFmpeg not available. Install via: pip install static-ffmpeg")

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
    print("FAST NEUFLOW PROCESSING (10-70x faster than RAFT)")
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
        return

    mask_files = sorted([f for f in os.listdir(MASKS_FOLDER) if f.endswith('.png')])
    print(f"Masks: {len(mask_files)} files")
    print()

    # Create temp dirs
    temp_prefix = f"neuflow_{Path(video_path).stem}"
    frames_dir = Path(TEMP_DIR) / f"{temp_prefix}_frames"
    cropped_dir = Path(TEMP_DIR) / f"{temp_prefix}_cropped"
    masks_dir = Path(TEMP_DIR) / f"{temp_prefix}_masks"
    output_dir = Path(TEMP_DIR) / f"{temp_prefix}_output"

    for path in [frames_dir, cropped_dir, masks_dir, output_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print("[1/6] Extracting frames...")
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
    print(f"\n[2/6] Loading masks...")
    import shutil
    for i, mask_file in enumerate(mask_files[:extracted_frames]):
        src = Path(MASKS_FOLDER) / mask_file
        dst = masks_dir / f"{i:04d}.png"
        shutil.copy2(src, dst)

    # Calculate crop region
    print(f"\n[3/6] Analyzing masks for crop region...")
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    for i in range(extracted_frames):
        mask_path = masks_dir / f"{i:04d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)

    bbox = [min_x, min_y, max_x, max_y]
    crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
    print(f"  Crop: {crop_x},{crop_y} {crop_w}x{crop_h}")

    # Crop frames and masks
    print(f"\n[4/6] Cropping...")
    for i in range(extracted_frames):
        frame_file = f"{i:04d}.png"

        # Crop frame
        frame = cv2.imread(str(frames_dir / frame_file))
        if frame is not None:
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            cv2.imwrite(str(cropped_dir / frame_file), cropped)

        # Crop mask
        mask_path = masks_dir / frame_file
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                cv2.imwrite(str(mask_path), cropped_mask)

    # Run ProPainter with NeuFlow TensorRT
    print(f"\n[5/6] Running ProPainter with NeuFlow TensorRT (FAST!)...")

    import torch
    use_fp16 = torch.cuda.is_available()

    faster_propainter_pipeline(
        video=str(cropped_dir),
        mask=str(masks_dir),
        output=str(output_dir),
        resize_ratio=1.0,
        mask_dilation=4,
        ref_stride=10,
        neighbor_length=10,
        subvideo_length=80,
        raft_iter=20,
        mode="video_inpainting",
        save_frames=True,
        fp16=use_fp16
    )

    # Find ProPainter output
    propainter_output = output_dir / cropped_dir.name / "frames"

    # Merge back to full resolution
    print(f"\n[6/6] Encoding final video...")
    final_frames_dir = output_dir / "final"
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

    # Encode with GPU NVENC (10-50x faster than CPU)
    output_path = Path(RESULT_DIR) / f"{Path(video_path).stem}_neuflow_cleaned.mp4"
    output_path.parent.mkdir(exist_ok=True)

    # Get FFmpeg path
    ffmpeg_exe, _ = get_ffmpeg_executables()

    print(f"  Using GPU NVENC encoding (RTX 4090)...")

    ffmpeg_cmd = [
        ffmpeg_exe,
        '-framerate', str(fps),
        '-i', str(final_frames_dir / '%04d.png'),
        '-i', video_path,
        '-map', '0:v:0',
        '-map', '1:a:0?',
        # GPU NVENC encoding (10-50x faster than CPU)
        '-c:v', 'h264_nvenc',           # NVIDIA GPU encoder
        '-preset', 'p7',                 # Highest quality preset (p1-p7)
        '-tune', 'hq',                   # High quality tuning
        '-rc', 'vbr',                    # Variable bitrate
        '-cq', '18',                     # Quality level (lower = better)
        '-b:v', '0',                     # Let encoder decide bitrate
        '-maxrate', '20M',               # Max bitrate cap
        '-bufsize', '40M',               # Buffer size
        '-profile:v', 'high',            # H.264 high profile
        '-level', '5.1',                 # H.264 level 5.1
        '-c:a', 'aac',                   # Audio codec
        '-b:a', '192k',                  # Audio bitrate
        '-pix_fmt', 'yuv420p',           # Pixel format
        '-y',
        str(output_path)
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] FFmpeg encoding failed:")
        print(result.stderr)
        # Try fallback to CPU encoding
        print(f"\n[FALLBACK] Trying CPU encoding instead...")
        ffmpeg_cmd_fallback = [
            ffmpeg_exe,
            '-framerate', str(fps),
            '-i', str(final_frames_dir / '%04d.png'),
            '-i', video_path,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_path)
        ]
        result = subprocess.run(ffmpeg_cmd_fallback, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] CPU encoding also failed: {result.stderr}")
            return

    print()
    print("="*80)
    print("✅ DONE!")
    print("="*80)
    print(f"Output: {output_path}")
    print("="*80)

if __name__ == "__main__":
    main()
    input("\nPress ENTER to exit...")
