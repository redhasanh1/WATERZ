#!/usr/bin/env python3
"""
Mask Overlay Viewer - View SAM2 masks overlaid on video
Usage: Double-click view_masks.bat or run: python view_masks.py

Controls:
  Space  - Pause/Resume
  Q/Esc  - Quit
  Left   - Previous frame
  Right  - Next frame
  [/]    - Decrease/Increase playback speed
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog


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
    print("Mask Overlay Viewer")
    print("=" * 60)

    # Select video
    print("\n[1/2] Select video file...")
    video_path = select_video()
    if not video_path:
        print("No video selected. Exiting.")
        return
    print(f"Video: {video_path}")

    # Select masks folder
    print("\n[2/2] Select masks folder...")
    masks_folder = select_masks_folder()
    if not masks_folder:
        print("No masks folder selected. Exiting.")
        return
    print(f"Masks: {masks_folder}")

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

    print("\n" + "=" * 60)
    print("Controls:")
    print("  Space  - Pause/Resume")
    print("  Q/Esc  - Quit")
    print("  Left   - Previous frame")
    print("  Right  - Next frame")
    print("  [/]    - Slower/Faster playback")
    print("=" * 60)

    # Playback state
    paused = False
    frame_idx = 0
    speed_multiplier = 1.0

    # Create window
    window_name = "Mask Overlay Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

    while True:
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame_idx = 0
            continue

        # Load and overlay mask
        mask = load_mask(masks_folder, frame_idx)
        display = overlay_mask(frame, mask, color=(0, 255, 0), alpha=0.4)

        # Add info text
        info = f"Frame: {frame_idx}/{total_frames-1} | Speed: {speed_multiplier:.1f}x | {'PAUSED' if paused else 'PLAYING'}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        if mask is not None:
            cv2.putText(display, "MASK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "NO MASK", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show frame
        cv2.imshow(window_name, display)

        # Calculate wait time
        wait_ms = int(1000 / fps / speed_multiplier) if not paused else 0
        wait_ms = max(1, wait_ms)

        # Handle keyboard input
        key = cv2.waitKey(wait_ms if not paused else 50) & 0xFF

        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord(' '):  # Space - toggle pause
            paused = not paused
        elif key == 81 or key == 2424832:  # Left arrow
            frame_idx = max(0, frame_idx - 1)
        elif key == 83 or key == 2555904:  # Right arrow
            frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord('['):  # Slower
            speed_multiplier = max(0.1, speed_multiplier - 0.1)
        elif key == ord(']'):  # Faster
            speed_multiplier = min(4.0, speed_multiplier + 0.1)

        # Auto-advance if not paused
        if not paused:
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = 0  # Loop

    cap.release()
    cv2.destroyAllWindows()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
