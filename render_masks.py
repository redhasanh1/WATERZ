#!/usr/bin/env python3
"""
Mask Overlay Renderer - Render SAM2 masks overlaid on video to output file
Usage: Double-click render_masks.bat or run: python render_masks.py
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm


def select_video():
    """Open file picker for video"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return video_path


def select_masks_folder():
    """Open folder picker for masks"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    folder_path = filedialog.askdirectory(
        title="Select Masks Folder (containing .png files)"
    )
    root.destroy()
    return folder_path


def select_output_file():
    """Open save file dialog for output video"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    output_path = filedialog.asksaveasfilename(
        title="Save Output Video As",
        defaultextension=".mp4",
        filetypes=[
            ("MP4 video", "*.mp4"),
            ("AVI video", "*.avi"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return output_path


def load_mask(masks_folder, frame_idx):
    """Load mask for given frame index"""
    mask_path = os.path.join(masks_folder, f"{frame_idx:05d}.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask
    return None


def overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4):
    """Overlay colored mask on frame"""
    if mask is None:
        return frame

    # Resize mask to match frame if needed
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create colored overlay
    overlay = frame.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = color

    # Blend
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    # Draw mask contours for better visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)

    return result


def main():
    print("=" * 60)
    print("Mask Overlay Renderer - Output to Video File")
    print("=" * 60)

    # Select video
    print("\n[1/3] Select video file...")
    video_path = select_video()
    if not video_path:
        print("No video selected. Exiting.")
        return
    print(f"Video: {video_path}")

    # Select masks folder
    print("\n[2/3] Select masks folder...")
    masks_folder = select_masks_folder()
    if not masks_folder:
        print("No masks folder selected. Exiting.")
        return
    print(f"Masks: {masks_folder}")

    # Select output file
    print("\n[3/3] Select output file location...")
    output_path = select_output_file()
    if not output_path:
        print("No output file selected. Exiting.")
        return
    print(f"Output: {output_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {width}x{height} @ {fps:.1f} fps, {total_frames} frames")

    # Count masks
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.png')]
    print(f"Masks: {len(mask_files)} files found")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nRendering overlay video...")

    # Process frames
    frame_idx = 0
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        # Load and overlay mask
        mask = load_mask(masks_folder, frame_idx)
        result = overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4)

        # Write frame
        out.write(result)

    cap.release()
    out.release()

    print(f"\n{'=' * 60}")
    print(f"Done! Output saved to: {output_path}")
    print(f"Frames rendered: {frame_idx + 1}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
