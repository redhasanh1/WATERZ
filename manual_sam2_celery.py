"""
Manual SAM2 Watermark Removal - NO YOLO
User draws bbox → SAM2 tracks → ProPainter removes
"""

import os
import sys
sys.path.insert(0, "faster-propainter-main")

import cv2
import numpy as np
from sam2_tracker import SAM2Tracker, manual_select_point
from watermark import pipeline


def manual_sam2_workflow(video_path, output_path):
    """
    Manual watermark removal workflow:
    1. User draws bbox on first frame
    2. SAM2-Tiny tracks across all frames (44ms/frame)
    3. ProPainter removes with your optimized pipeline

    NO YOLO - 100% manual control
    """
    print("="*80)
    print("MANUAL SAM2 WATERMARK REMOVAL (NO YOLO)")
    print("="*80)

    # Load video
    print("\n[1/4] Loading video...")
    cap = cv2.VideoCapture(video_path)

    frames_bgr = []
    frames_rgb = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    print(f"[OK] {len(frames_rgb)} frames loaded ({frames_rgb[0].shape[1]}x{frames_rgb[0].shape[0]} @ {fps} fps)")

    # Initialize SAM2 tracker first (needed for preview)
    print("\n[2/4] Initializing SAM2-Tiny...")
    tracker = SAM2Tracker(device="cuda")

    # Manual point selection with LIVE PREVIEW
    print("\n[3/4] MANUAL SELECTION - Click on watermark...")
    point = manual_select_point(frames_rgb[0], tracker=tracker, video_path=video_path)

    if point is None:
        print("[ERROR] No selection made, exiting")
        return

    print(f"[OK] Confirmed point: {point}")

    # SAM2 tracking across all frames
    print("\n[4/4] SAM2-Tiny tracking with temporal consistency...")

    masks_bool = tracker.track_from_points(
        video_frames=frames_rgb,
        points=[point],
        labels=[1],  # 1 = positive point (foreground)
        frame_idx=0,
        video_path=video_path
    )

    # Convert to uint8
    masks_uint8 = [(mask * 255).astype(np.uint8) for mask in masks_bool]
    print(f"[OK] {len(masks_uint8)} masks generated")

    # ProPainter inpainting
    print("\n[5/5] ProPainter inpainting (your optimized pipeline)...")
    pipeline(
        video=video_path,
        mask=None,
        output=output_path,
        frames_array=np.array(frames_bgr),
        masks_array=np.array(masks_uint8),
        fp16=True,
        save_fps=fps,
        subvideo_length=40,
        neighbor_length=10
    )

    print("\n" + "="*80)
    print("[SUCCESS] Watermark removed!")
    print(f"[OUTPUT] {output_path}")
    print("="*80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python manual_sam2_celery.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./output_manual_sam2"

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    manual_sam2_workflow(video_path, output_path)
