"""
SAM2 Mask Visualization Tool

Creates a visual debug video showing:
- Original frame
- SAM2 mask
- Overlay (mask on frame with transparency)

Helps diagnose SAM2 tracking quality issues.
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Configuration
MASKS_FOLDER = "temp_sam2_masks"
VIDEO_PATH = "test_3sec_16fps.mp4"
OUTPUT_VIDEO = "results/sam2_mask_visualization.mp4"

def visualize_masks():
    """Create visualization video showing masks overlaid on frames."""

    print("\n" + "="*70)
    print("SAM2 MASK VISUALIZATION")
    print("="*70 + "\n")

    if not os.path.exists(MASKS_FOLDER):
        print(f"❌ ERROR: Masks folder not found: {MASKS_FOLDER}")
        print("   Run the SAM2 interactive tool first to generate masks.")
        return

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video: {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video: {VIDEO_PATH}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print()

    # Get mask files
    mask_files = sorted([f for f in os.listdir(MASKS_FOLDER) if f.endswith('.png')])

    if len(mask_files) == 0:
        print(f"❌ ERROR: No mask files found in {MASKS_FOLDER}")
        return

    print(f"🎭 Found {len(mask_files)} mask files")
    print()

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Create video writer (triple width for side-by-side)
    output_width = width * 3
    output_height = height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (output_width, output_height))

    if not out.isOpened():
        print(f"❌ ERROR: Could not create output video: {OUTPUT_VIDEO}")
        return

    print("🎬 Creating visualization video...")
    print(f"   Output: {OUTPUT_VIDEO}")
    print()

    frame_idx = 0
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Load corresponding mask
        if frame_idx < len(mask_files):
            mask_path = os.path.join(MASKS_FOLDER, mask_files[frame_idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None

        # Create visualization
        if mask is not None and mask.shape[:2] == (height, width):
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

            # Add labels
            cv2.putText(frame, f"Frame {frame_idx}", (10, 30), font, font_scale, color, thickness)
            cv2.putText(mask_color, f"Mask ({coverage_pct:.2f}%)", (10, 30), font, font_scale, color, thickness)
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
        frame_idx += 1

        # Progress indicator
        if frames_processed % 10 == 0 or frames_processed == total_frames:
            print(f"   Processed {frames_processed}/{total_frames} frames...", end='\r')

    print(f"\n\n✅ Visualization complete!")
    print(f"   Output: {OUTPUT_VIDEO}")
    print(f"   Processed {frames_processed} frames")
    print()

    print("📺 View the output video to see:")
    print("   - Left: Original frame")
    print("   - Middle: SAM2 mask (white = object)")
    print("   - Right: Overlay with red mask + green bounding box")
    print()

    cap.release()
    out.release()

    # Open the video automatically (Windows)
    try:
        print("🎥 Opening visualization video...")
        os.startfile(OUTPUT_VIDEO)
    except:
        print(f"   Please open manually: {OUTPUT_VIDEO}")

    print("\n" + "="*70)
    print()

if __name__ == "__main__":
    visualize_masks()
