"""
SAM2 PyTorch Mask Generation (Old Method - Temporal Propagation)
Uses the original PyTorch SAM2 tracker from 3 commits ago
Shows IoU scores, temporal consistency metrics, and saves masks
"""

import os
import sys
import cv2
import numpy as np
import time
import torch
import shutil
from pathlib import Path
from tkinter import Tk, filedialog

# Add SAM2 to path
SAM2_PATH = Path(__file__).parent / "segment-anything-2"
sys.path.insert(0, str(SAM2_PATH))

from sam2.build_sam import build_sam2_video_predictor

# Configuration
MASKS_OUTPUT = "temp_sam2_masks"


class SAM2Tracker:
    """SAM2-Tiny video tracker with temporal propagation"""

    def __init__(self, device="cuda"):
        self.device = device

        # Paths
        checkpoint_dir = SAM2_PATH / "checkpoints"
        self.checkpoint = str(checkpoint_dir / "sam2.1_hiera_tiny.pt")
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint}")

        print(f"[SAM2] Loading PyTorch model...")
        print(f"[SAM2] Checkpoint: {self.checkpoint}")
        print(f"[SAM2] Device: {device}")

        # Build video predictor
        self.predictor = build_sam2_video_predictor(
            self.model_cfg,
            self.checkpoint,
            device=device
        )

        print(f"[SAM2] Model loaded! (38.9M params)")

    def track_from_points(self, video_path, points, labels, frame_idx=0, object_id=1):
        """
        Track object from point prompts with temporal propagation

        Returns:
            masks: numpy array [T, H, W] with binary masks
            video_segments: dict with per-frame segmentation data
            inference_state: SAM2 inference state
        """
        print(f"\n[SAM2] Starting temporal propagation tracking...")

        # Initialize inference state from video
        print(f"[SAM2] Loading video: {video_path}")
        inference_state = self.predictor.init_state(video_path=video_path)
        num_frames = len(inference_state["images"])
        h = inference_state["video_height"]
        w = inference_state["video_width"]
        print(f"[SAM2] Loaded {num_frames} frames ({w}x{h})")

        # Add point prompts on first frame
        print(f"[SAM2] Adding {len(points)} point(s) on frame {frame_idx}...")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=np.array(points, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32)
        )

        # Propagate with temporal consistency
        print(f"[SAM2] Propagating masks across {num_frames} frames with temporal consistency...")
        video_segments = {}

        start_time = time.time()
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        total_time = time.time() - start_time

        # Extract masks
        masks = []
        for frame_idx in sorted(video_segments.keys()):
            if object_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][object_id][0]  # [H, W]
                masks.append(mask)
            else:
                masks.append(np.zeros((h, w), dtype=bool))

        masks = np.array(masks)  # [T, H, W]

        print(f"[SAM2] Tracking complete!")
        print(f"[SAM2] Time: {total_time:.3f}s ({total_time/num_frames*1000:.1f}ms per frame)")
        print(f"[SAM2] Masks: {masks.shape}")
        print(f"[SAM2] Coverage: {masks.sum() / masks.size * 100:.2f}% of pixels")

        return masks, video_segments, inference_state


def select_video_file():
    """Open file dialog to select video"""
    print("\n[SELECT] Please choose a video file...")

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select Video for SAM2 PyTorch Mask Generation",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        print("[ERROR] No file selected")
        sys.exit(1)

    print(f"[OK] Selected: {file_path}\n")
    return file_path


def select_point_interactive(video_path, tracker):
    """Interactive point selection with live SAM2 preview"""
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read first frame")
        sys.exit(1)

    frame_bgr = first_frame.copy()
    frame_original = frame_bgr.copy()

    selected_point = [None, None]
    preview_mask = None

    def show_mask_preview(point):
        """Generate SAM2 mask preview"""
        print(f"[SAM2] Generating mask preview for point {point}...")

        inference_state = tracker.predictor.init_state(video_path=video_path)

        _, out_obj_ids, out_mask_logits = tracker.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=np.array([point], dtype=np.float32),
            labels=np.array([1], dtype=np.int32)
        )

        mask = (out_mask_logits[0][0] > 0.0).cpu().numpy()
        print(f"[SAM2] Preview coverage: {mask.sum() / mask.size * 100:.2f}%")
        return mask

    def mouse_callback(event, x, y, flags, param):
        nonlocal preview_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point[0] = x
            selected_point[1] = y

            preview_mask = show_mask_preview([x, y])

            display = frame_original.copy()

            if preview_mask is not None:
                overlay = display.copy()
                overlay[preview_mask] = [0, 255, 0]  # Green
                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, "SPACE/ENTER = Confirm | ESC = Cancel | Click = New point",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("SAM2 PyTorch: Click on Object", display)

    print("\n[INTERACTIVE] Click on the object to track:")
    print("  - Click to see GREEN MASK PREVIEW")
    print("  - Press SPACE/ENTER to confirm")
    print("  - Press ESC to cancel")

    cv2.namedWindow("SAM2 PyTorch: Click on Object")
    cv2.setMouseCallback("SAM2 PyTorch: Click on Object", mouse_callback)
    cv2.imshow("SAM2 PyTorch: Click on Object", frame_bgr)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("[CANCELLED]")
            sys.exit(0)
        elif key in [13, 32]:  # ENTER or SPACE
            if selected_point[0] is not None:
                cv2.destroyAllWindows()
                print(f"[OK] Confirmed point: {selected_point}")
                return selected_point


def calculate_metrics(masks):
    """Calculate temporal consistency and quality metrics"""
    print(f"\n{'='*70}")
    print(f"MASK QUALITY METRICS (PyTorch SAM2 Temporal Propagation)")
    print(f"{'='*70}")

    # Frame-to-frame IoU (temporal consistency)
    ious = []
    for i in range(len(masks) - 1):
        mask_curr = masks[i]
        mask_next = masks[i + 1]

        intersection = np.logical_and(mask_curr, mask_next).sum()
        union = np.logical_or(mask_curr, mask_next).sum()
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)

    # Mask area statistics
    areas = [mask.sum() for mask in masks]
    avg_area = np.mean(areas)
    area_std = np.std(areas)
    area_stability = 1.0 - (area_std / avg_area) if avg_area > 0 else 0.0

    # Print results
    print(f"\nTemporal Consistency (Frame-to-Frame IoU):")
    print(f"  Average IoU: {np.mean(ious):.4f} (higher = more stable)")
    print(f"  Min IoU: {np.min(ious):.4f}")
    print(f"  Max IoU: {np.max(ious):.4f}")
    print(f"  Std Dev: {np.std(ious):.4f}")
    print()
    print(f"Mask Area Stability:")
    print(f"  Average Area: {avg_area:.1f} pixels")
    print(f"  Std Dev: {area_std:.1f} pixels")
    print(f"  Stability Score: {area_stability:.4f} (1.0 = perfect)")
    print()

    # Identify problematic frames
    low_iou_threshold = np.mean(ious) - 2 * np.std(ious)
    low_iou_frames = [(i+1, ious[i]) for i in range(len(ious)) if ious[i] < low_iou_threshold]

    if low_iou_frames:
        print(f"Frames with Low IoU (< {low_iou_threshold:.4f}):")
        for frame_idx, iou in low_iou_frames[:10]:
            print(f"  Frame {frame_idx:05d}: IoU = {iou:.4f}")
        if len(low_iou_frames) > 10:
            print(f"  ... and {len(low_iou_frames) - 10} more")
    else:
        print(f"✓ All frames have good temporal consistency!")

    print(f"{'='*70}")


def save_masks(masks, output_dir):
    """Save masks as PNG images"""
    print(f"\n[SAVING] Saving masks to {output_dir}...")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/{i:05d}.png", mask_uint8)

    print(f"[OK] Saved {len(masks)} masks")


def main():
    print("="*70)
    print("SAM2 PYTORCH MASK GENERATION")
    print("(Old Method - Temporal Propagation)")
    print("="*70)

    # Select video
    video_path = select_video_file()

    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"\n[VIDEO] {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Initialize tracker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = SAM2Tracker(device=device)

    # Select point with preview
    point = select_point_interactive(video_path, tracker)

    # Generate masks with temporal propagation
    print(f"\n{'='*70}")
    print(f"PROCESSING")
    print(f"{'='*70}")

    start_total = time.time()
    masks, video_segments, inference_state = tracker.track_from_points(
        video_path=video_path,
        points=[point],
        labels=[1],
        frame_idx=0
    )
    total_time = time.time() - start_total

    # Calculate metrics
    calculate_metrics(masks)

    # Performance summary
    print(f"\n{'='*70}")
    print(f"PERFORMANCE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.3f}s")
    print(f"FPS: {len(masks)/total_time:.2f}")
    print(f"Avg time/frame: {total_time/len(masks)*1000:.1f}ms")
    print(f"{'='*70}")

    # Save masks
    save_masks(masks, MASKS_OUTPUT)

    print(f"\n{'='*70}")
    print(f"COMPLETE!")
    print(f"{'='*70}")
    print(f"")
    print(f"Masks saved to: {MASKS_OUTPUT}/")
    print(f"")
    print(f"Next steps:")
    print(f"  - Run RUN_VISUALIZE_SAM2_MASKS.bat to view masks")
    print(f"")


if __name__ == "__main__":
    main()
