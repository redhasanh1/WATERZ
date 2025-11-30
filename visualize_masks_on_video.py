"""
SAM2 Mask Visualization - Overlay 10fps masks on video with proper expansion
Usage: python visualize_masks_on_video.py [video_path]

Reads mask_mapping.json to expand 10fps masks to original video FPS.
"""
import os
import sys
import cv2
import json
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

def load_mask_mapping(masks_dir):
    """Load mask mapping info for 10fps -> original FPS expansion"""
    mapping_path = masks_dir / "mask_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            return json.load(f)
    return None

def main():
    print("="*60)
    print("SAM2 Mask Visualization - With 10fps Expansion")
    print("="*60)

    # Parse args
    video_path = None
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]

    if not video_path or not os.path.exists(video_path):
        video_path = select_video()
        if not video_path:
            print("[CANCELLED]")
            return

    # Paths
    masks_dir = Path("temp_sam2_masks")
    output_path = Path("results") / "mask_visualization.mp4"

    if not masks_dir.exists():
        print(f"[ERROR] Masks folder not found: {masks_dir}")
        return

    # Load mask mapping for 10fps expansion
    mapping = load_mask_mapping(masks_dir)

    if mapping:
        multiplier = mapping['multiplier']
        original_fps = mapping['original_fps']
        total_frames = mapping['total_frames']
        num_masks = mapping['num_masks']
        print(f"[10FPS MODE] Found mask_mapping.json")
        print(f"  Original FPS: {original_fps}")
        print(f"  Multiplier: {multiplier}x (each mask repeats {int(multiplier)} times)")
        print(f"  Masks: {num_masks} -> Frames: {total_frames}")
    else:
        multiplier = 1.0
        original_fps = 30.0
        print("[1:1 MODE] No mask_mapping.json - using direct mapping")

    # Get mask files
    mask_files = sorted(masks_dir.glob("*.png"))
    mask_files = [f for f in mask_files if f.name != "mask_mapping.json"]

    print(f"[OK] Found {len(mask_files)} mask files")

    if len(mask_files) == 0:
        print("[ERROR] No masks found!")
        return

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[VIDEO] {video_path}")
    print(f"  Frames: {total_video_frames}, FPS: {video_fps:.2f}, Size: {w}x{h}")

    # Create output dir
    output_path.parent.mkdir(exist_ok=True)

    # Create video writer at ORIGINAL fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (w, h))

    print(f"\n[PROCESSING] Creating visualization at {video_fps:.1f}fps...")
    print(f"  Each mask will be used for ~{multiplier:.1f} frames")

    # Pre-load all masks for speed
    print("[LOADING] Pre-loading masks into memory...")
    masks_cache = {}
    for mask_path in mask_files:
        idx = int(mask_path.stem)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # Resize if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masks_cache[idx] = mask
    print(f"[OK] Loaded {len(masks_cache)} masks")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate which mask to use (10fps expansion)
        # mask_idx = frame_idx // multiplier
        mask_idx = int(frame_idx / multiplier)
        mask_idx = min(mask_idx, len(mask_files) - 1)  # Clamp to available masks

        if mask_idx in masks_cache:
            mask = masks_cache[mask_idx]

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
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_video_frames} (using mask {mask_idx})")

    cap.release()
    out.release()

    print(f"\n[DONE] Saved to: {output_path}")
    print(f"  Output: {frame_idx} frames at {video_fps:.1f}fps")
    print(f"\nOpening video...")
    os.startfile(str(output_path))

if __name__ == "__main__":
    main()
