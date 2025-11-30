"""
Simple script to overlay SAM2 masks on video frames
Usage: python visualize_masks_on_video.py
"""
import os
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog

def select_video():
    """Open file picker"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select Video File",
        initialdir=os.getcwd(),
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

def main():
    print("="*60)
    print("SAM2 Mask Visualization - Overlay masks on video")
    print("="*60)

    # Paths
    frames_dir = Path("temp_sam2_frames")
    masks_dir = Path("temp_sam2_masks")
    output_path = Path("results") / "mask_visualization.mp4"

    # Check folders exist
    if not frames_dir.exists():
        print(f"[ERROR] Frames folder not found: {frames_dir}")
        return
    if not masks_dir.exists():
        print(f"[ERROR] Masks folder not found: {masks_dir}")
        return

    # Get frame files
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.png"))

    mask_files = sorted(masks_dir.glob("*.png"))

    print(f"[OK] Found {len(frame_files)} frames")
    print(f"[OK] Found {len(mask_files)} masks")

    if len(frame_files) == 0:
        print("[ERROR] No frames found!")
        return

    # Create output dir
    output_path.parent.mkdir(exist_ok=True)

    # Get video dimensions from first frame
    first_frame = cv2.imread(str(frame_files[0]))
    h, w = first_frame.shape[:2]
    print(f"[OK] Frame size: {w}x{h}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"\n[PROCESSING] Creating visualization video...")

    for i, frame_path in enumerate(frame_files):
        # Read frame
        frame = cv2.imread(str(frame_path))

        # Find matching mask
        mask_path = masks_dir / f"{i:05d}.png"

        if mask_path.exists():
            # Read mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Resize mask if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Create overlay
            mask_bool = mask > 127

            # Green overlay
            overlay = frame.copy()
            overlay[mask_bool, 1] = np.clip(overlay[mask_bool, 1].astype(int) + 100, 0, 255)

            # Blend
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Write frame
        out.write(frame)

        if i % 50 == 0:
            print(f"  Frame {i}/{len(frame_files)}")

    out.release()

    print(f"\n[DONE] Saved to: {output_path}")
    print(f"\nOpening video...")
    os.startfile(str(output_path))

if __name__ == "__main__":
    main()
