"""
SAM2 Mask Visualization Tool - 10 FPS MODE

Creates a visual debug video showing:
- Original frame
- SAM2 mask
- Overlay (mask on frame with transparency)

**10 FPS MODE**: Each mask is repeated 10 times
Example: If you have 10 masks, this creates 100 frames total

Uses EXISTING masks from temp_sam2_masks folder.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
import subprocess

# Configuration (can be overridden by command line args)
MASKS_FOLDER = "temp_sam2_masks"
OUTPUT_VIDEO = "results/sam2_mask_visualization_10fps.mp4"
MASK_REPEAT_COUNT = 10  # Each mask is used for 10 frames
VIDEO_PATH_ARG = None  # Will be set from command line

def select_video_file():
    """Open file dialog to select video file."""
    print("\n[*] Please select the video file you want to visualize...")

    root = Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front

    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        print("[ERROR] No file selected")
        return None

    print(f"[OK] Selected: {file_path}\n")
    return file_path

def visualize_masks():
    """Create visualization video showing masks overlaid on frames - 10fps mode."""

    print("\n" + "="*70)
    print("SAM2 MASK VISUALIZATION - 10 FPS MODE")
    print("="*70 + "\n")
    print(f"Each mask will be repeated {MASK_REPEAT_COUNT} times")
    print()

    # Check if masks exist
    if not os.path.exists(MASKS_FOLDER):
        print(f"[ERROR] Masks folder not found: {MASKS_FOLDER}")
        print("   Please run the SAM2 interactive tool first to generate masks.")
        print("   The tool should create masks in the temp_sam2_masks folder.")
        return

    mask_files = sorted([f for f in os.listdir(MASKS_FOLDER) if f.endswith('.png')])
    if len(mask_files) == 0:
        print(f"[ERROR] No mask files found in {MASKS_FOLDER}")
        print("   Please run the SAM2 interactive tool first to generate masks.")
        return

    print(f"[MASKS] Found {len(mask_files)} mask files in: {MASKS_FOLDER}")
    print()

    # PRE-LOAD all masks into RAM for instant access (much faster than reading from disk each frame)
    print(f"[PRELOAD] Loading {len(mask_files)} masks into RAM from NVME...")
    import time
    start_time = time.time()
    preloaded_masks = []
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(MASKS_FOLDER, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        preloaded_masks.append(mask)
        if (i + 1) % 1000 == 0:
            print(f"   Loaded {i + 1}/{len(mask_files)} masks...", end='\r')
    elapsed = time.time() - start_time
    print(f"[PRELOAD] ✓ All {len(mask_files)} masks loaded in {elapsed:.1f}s ({len(mask_files)/elapsed:.0f} masks/sec)")
    print()

    # Interactive video selection (or use command line arg)
    if VIDEO_PATH_ARG:
        VIDEO_PATH = VIDEO_PATH_ARG
        print(f"[VIDEO] Using video from command line: {VIDEO_PATH}\n")
    else:
        VIDEO_PATH = select_video_file()
        if VIDEO_PATH is None:
            return

    # Get video info first (need dimensions for FFmpeg)
    cap_info = cv2.VideoCapture(VIDEO_PATH)
    if not cap_info.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_PATH}")
        return

    width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_info.release()

    # Get FFmpeg from static-ffmpeg for GPU decoding
    import static_ffmpeg
    ffmpeg_path_decode, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()

    # Start FFmpeg with NVDEC GPU hardware decoding
    ffmpeg_decode_cmd = [
        ffmpeg_path_decode,
        '-hwaccel', 'cuda',          # Enable CUDA hardware acceleration
        '-i', VIDEO_PATH,             # Input video
        '-f', 'rawvideo',             # Raw pixel output
        '-pix_fmt', 'bgr24',          # BGR format for OpenCV compatibility
        '-vsync', '0',                # No frame duplication
        '-'                           # Output to stdout pipe
    ]

    print("[GPU] Starting NVDEC hardware video decoding...")
    ffmpeg_decode_process = subprocess.Popen(
        ffmpeg_decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=width * height * 3 * 10  # Buffer 10 frames
    )

    print(f"[VIDEO] Video: {VIDEO_PATH}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total input frames: {total_frames}")
    print()

    # **CALCULATE** repeat count based on video frames and available masks
    calculated_repeat = total_frames / len(mask_files)
    print(f"[10FPS] Auto-calculating repeat count:")
    print(f"   Video frames: {total_frames}")
    print(f"   Available masks: {len(mask_files)}")
    print(f"   Repeat each mask: {calculated_repeat:.1f} times")
    print()

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Create video writer (triple width for side-by-side) - GPU NVENC encoding
    output_width = width * 3
    output_height = height

    # Get FFmpeg from static-ffmpeg
    import static_ffmpeg
    ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()

    # Start FFmpeg process with NVENC GPU encoding
    ffmpeg_cmd = [
        ffmpeg_path,
        '-y',  # Overwrite output
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{output_width}x{output_height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # Input from pipe
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',  # Fast preset
        '-b:v', '10M',
        '-pix_fmt', 'yuv420p',
        OUTPUT_VIDEO
    ]

    print("[GPU] Using NVENC hardware encoding for maximum speed!")
    print("[PROCESSING] Creating visualization video (10fps mode)...")
    print(f"   Output: {OUTPUT_VIDEO}")
    print()

    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_idx = 0
    frames_processed = 0
    output_frames_generated = 0
    frame_size = width * height * 3  # BGR24 = 3 bytes per pixel

    while True:
        # Read raw frame from GPU decode pipe
        raw_frame = ffmpeg_decode_process.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) != frame_size:
            break

        # Convert to numpy array and make writable copy (frombuffer is read-only)
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()

        # **KEY CHANGE**: Calculate mask index by dividing frame_idx by calculated repeat count
        # This spreads the available masks across all video frames
        mask_idx = int(frame_idx / calculated_repeat)

        # Get pre-loaded mask (instant - no disk I/O!)
        if mask_idx < len(preloaded_masks):
            mask = preloaded_masks[mask_idx]

            # Resize mask to match video dimensions if needed
            if mask is not None and mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            mask = None

        # FAST GPU MODE: Simple side-by-side visualization (no text overlays)
        if mask is not None:
            # Convert mask to 3-channel (fast operation)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Create simple red overlay (fast operation)
            red_mask = np.zeros_like(frame)
            red_mask[:, :, 2] = mask  # Red channel only
            overlay = cv2.addWeighted(frame, 0.7, red_mask, 0.3, 0)

            # Combine side-by-side (fast operation)
            combined = np.hstack([frame, mask_color, overlay])
        else:
            # No mask - show black frames
            no_mask = np.zeros_like(frame)
            combined = np.hstack([frame, no_mask, no_mask])

        # Write frame to FFmpeg pipe
        ffmpeg_process.stdin.write(combined.tobytes())
        frames_processed += 1
        output_frames_generated += 1
        frame_idx += 1

        # Progress indicator
        if frames_processed % 10 == 0 or frames_processed == total_frames:
            print(f"   Processed {frames_processed}/{total_frames} video frames (Output: {output_frames_generated} frames)...", end='\r')

    print(f"\n\n[OK] Visualization complete!")
    print(f"   Output: {OUTPUT_VIDEO}")
    print(f"   Input frames: {frames_processed}")
    print(f"   Output frames: {output_frames_generated}")
    print(f"   Masks used: {len(mask_files)}")
    print(f"   Each mask repeated: ~{int(calculated_repeat)} times")
    print()

    print("[INFO] View the output video to see:")
    print("   - Left: Original frame")
    print("   - Middle: SAM2 mask (white = object) - shows mask # and repeat count")
    print("   - Right: Overlay with red mask + green bounding box")
    print()

    # Close FFmpeg decode process
    ffmpeg_decode_process.stdout.close()
    ffmpeg_decode_process.wait()

    # Close FFmpeg encode process
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    # Open the video automatically (Windows)
    try:
        print("[VIDEO] Opening visualization video...")
        os.startfile(OUTPUT_VIDEO)
    except:
        print(f"   Please open manually: {OUTPUT_VIDEO}")

    print("\n" + "="*70)
    print()

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) >= 2:
        VIDEO_PATH_ARG = sys.argv[1]
    if len(sys.argv) >= 3:
        MASKS_FOLDER = sys.argv[2]

    visualize_masks()
