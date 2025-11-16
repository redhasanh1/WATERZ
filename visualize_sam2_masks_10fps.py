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

# Configuration
MASKS_FOLDER = "temp_sam2_masks"
OUTPUT_VIDEO = "results/sam2_mask_visualization_10fps.mp4"
MASK_REPEAT_COUNT = 10  # Each mask is used for 10 frames

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

    # Interactive video selection
    VIDEO_PATH = select_video_file()
    if VIDEO_PATH is None:
        return

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    # Create video writer (triple width for side-by-side)
    output_width = width * 3
    output_height = height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (output_width, output_height))

    if not out.isOpened():
        print(f"[ERROR] ERROR: Could not create output video: {OUTPUT_VIDEO}")
        return

    print("[PROCESSING] Creating visualization video (10fps mode)...")
    print(f"   Output: {OUTPUT_VIDEO}")
    print()

    frame_idx = 0
    frames_processed = 0
    output_frames_generated = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # **KEY CHANGE**: Calculate mask index by dividing frame_idx by calculated repeat count
        # This spreads the available masks across all video frames
        mask_idx = int(frame_idx / calculated_repeat)

        # Load corresponding mask (same mask for 10 consecutive frames)
        if mask_idx < len(mask_files):
            mask_path = os.path.join(MASKS_FOLDER, mask_files[mask_idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Resize mask to match video dimensions if needed
            if mask is not None and mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            mask = None

        # Create visualization
        if mask is not None:
            # Convert mask to 3-channel for visualization
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Create colored overlay (red mask with 50% transparency)
            overlay = frame.copy()
            red_mask = np.zeros_like(frame)
            red_mask[:, :, 2] = mask  # Red channel only
            overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (255, 255, 255)

            # Calculate mask coverage
            white_pixels = np.sum(mask > 127)
            total_pixels = width * height
            coverage_pct = (white_pixels / total_pixels) * 100

            # Add labels with mask index info
            repeat_num = int((frame_idx % calculated_repeat) + 1)
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30), font, font_scale, color, thickness)
            cv2.putText(mask_color, f"Mask #{mask_idx} ({coverage_pct:.2f}%)", (10, 30), font, font_scale, color, thickness)
            cv2.putText(mask_color, f"Repeat {repeat_num}/{int(calculated_repeat)}", (10, 60), font, 0.5, (255, 255, 0), 1)
            cv2.putText(overlay, f"Overlay", (10, 30), font, font_scale, color, thickness)

            # Get bounding box
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                # Draw bounding box on overlay
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Draw bbox info
                cv2.putText(overlay, f"BBox: {w}x{h}", (x, y-10), font, 0.5, (0, 255, 0), 1)

            # Combine side-by-side
            combined = np.hstack([frame, mask_color, overlay])
        else:
            # No mask available - show just the frame
            no_mask = np.zeros_like(frame)
            text_img = frame.copy()

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(no_mask, "NO MASK", (width//2 - 100, height//2), font, 1.5, (0, 0, 255), 3)
            cv2.putText(text_img, "NO MASK", (width//2 - 100, height//2), font, 1.5, (0, 0, 255), 3)

            combined = np.hstack([frame, no_mask, text_img])

        # Write frame
        out.write(combined)
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

    cap.release()
    out.release()

    # Open the video automatically (Windows)
    try:
        print("[VIDEO] Opening visualization video...")
        os.startfile(OUTPUT_VIDEO)
    except:
        print(f"   Please open manually: {OUTPUT_VIDEO}")

    print("\n" + "="*70)
    print()

if __name__ == "__main__":
    visualize_masks()
