"""
SAM2-Tiny Video Tracker for Watermark Removal
Provides temporally-consistent mask tracking across video frames for ProPainter

Features:
- Manual bounding box/point selection on first frame
- SAM2.1-Tiny model (FASTEST! 38.9M params, ~60-80ms/frame)
- Temporal consistency across all frames
- ProPainter-compatible mask format
"""

import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

# Add SAM2 to path
SAM2_PATH = Path(__file__).parent.parent / "segment-anything-2"
sys.path.insert(0, str(SAM2_PATH))

from sam2.build_sam import build_sam2_video_predictor


class SAM2Tracker:
    """
    SAM2-Tiny video tracker for fast, temporally-consistent segmentation
    """

    def __init__(self, device="cuda"):
        """
        Initialize SAM2-Tiny tracker

        Args:
            device: 'cuda' or 'cpu' (default: 'cuda')
        """
        self.device = device

        # Paths to SAM2.1-Tiny checkpoint and config
        checkpoint_dir = SAM2_PATH / "checkpoints"
        self.checkpoint = str(checkpoint_dir / "sam2.1_hiera_tiny.pt")
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

        # Check if checkpoint exists
        if not Path(self.checkpoint).exists():
            raise FileNotFoundError(
                f"SAM2.1-Tiny checkpoint not found at: {self.checkpoint}\n"
                f"Please download it from: "
                f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
            )

        print(f"[SAM2-Tiny] Loading model from: {self.checkpoint}")
        print(f"[SAM2-Tiny] Config: {self.model_cfg}")
        print(f"[SAM2-Tiny] Device: {device}")

        # Build SAM2 video predictor
        self.predictor = build_sam2_video_predictor(
            self.model_cfg,
            self.checkpoint,
            device=device
        )

        print("[SAM2-Tiny] Model loaded successfully!")
        print("[SAM2-Tiny] Ready for video tracking (38.9M params, ~60-80ms/frame)")

    def track_from_box(self, video_frames, bbox, frame_idx=0, object_id=1, video_path=None):
        """
        Track object from bounding box across all video frames

        Args:
            video_frames: List of numpy arrays [H, W, 3] (RGB format) OR video file path
            bbox: Bounding box [x1, y1, x2, y2] on frame_idx
            frame_idx: Frame index where bbox is specified (default: 0 = first frame)
            object_id: Unique ID for this object (default: 1)
            video_path: Optional video file path (if not using frames directly)

        Returns:
            masks: numpy array [T, H, W] with binary masks for all frames
        """
        print(f"\n[SAM2-Tiny] Starting video tracking...")

        # Use video_path for SAM2's built-in video loading
        if video_path is not None:
            print(f"[SAM2-Tiny] Loading video from: {video_path}")
            inference_state = self.predictor.init_state(video_path=video_path)
            num_frames = len(inference_state["images"])
            print(f"[SAM2-Tiny] Loaded {num_frames} frames from video")
        else:
            print(f"[SAM2-Tiny] ERROR: video_path required for SAM2")
            print(f"[SAM2-Tiny] Please pass video_path parameter")
            raise ValueError("video_path required - SAM2 needs video file path")

        print(f"[SAM2-Tiny] Bounding box: {bbox} on frame {frame_idx}")

        # Add bounding box prompt on specified frame
        print(f"[SAM2-Tiny] Adding bounding box prompt on frame {frame_idx}...")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            box=np.array(bbox, dtype=np.float32)
        )

        print(f"[SAM2-Tiny] Propagating masks across all {len(video_frames)} frames...")
        print(f"[SAM2-Tiny] Using temporal consistency for smooth tracking...")

        # Propagate masks to all frames with temporal consistency
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Extract masks for our object
        # Get dimensions from first mask or from inference state
        h = inference_state["video_height"]
        w = inference_state["video_width"]

        masks = []
        for frame_idx in sorted(video_segments.keys()):
            if object_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][object_id][0]  # [H, W]
                masks.append(mask)
            else:
                # Object not detected in this frame, add empty mask
                masks.append(np.zeros((h, w), dtype=bool))

        masks = np.array(masks)  # [T, H, W]

        print(f"[SAM2-Tiny] Tracking complete!")
        print(f"[SAM2-Tiny] Generated masks: {masks.shape}")
        print(f"[SAM2-Tiny] Mask coverage: {masks.sum() / masks.size * 100:.2f}% of pixels")

        return masks

    def track_from_points(self, video_frames, points, labels, frame_idx=0, object_id=1, video_path=None):
        """
        Track object from point prompts across all video frames

        Args:
            video_frames: List of numpy arrays [H, W, 3] (RGB format) OR video file path
            points: numpy array [[x1, y1], [x2, y2], ...] of prompt points
            labels: numpy array [1, 1, ...] (1=positive, 0=negative)
            frame_idx: Frame index where points are specified (default: 0)
            object_id: Unique ID for this object (default: 1)
            video_path: Optional video file path (if not using frames directly)

        Returns:
            masks: numpy array [T, H, W] with binary masks for all frames
        """
        print(f"\n[SAM2-Tiny] Starting video tracking from points...")

        # Use video_path for SAM2's built-in video loading
        if video_path is not None:
            print(f"[SAM2-Tiny] Loading video from: {video_path}")
            inference_state = self.predictor.init_state(video_path=video_path)
            num_frames = len(inference_state["images"])
            print(f"[SAM2-Tiny] Loaded {num_frames} frames from video")
        else:
            print(f"[SAM2-Tiny] ERROR: video_path required for SAM2")
            print(f"[SAM2-Tiny] Please pass video_path parameter")
            raise ValueError("video_path required - SAM2 needs video file path")

        print(f"[SAM2-Tiny] Points: {len(points)} on frame {frame_idx}")

        # Add point prompts
        print(f"[SAM2-Tiny] Adding point prompts on frame {frame_idx}...")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=np.array(points, dtype=np.float32),
            labels=np.array(labels, dtype=np.int32)
        )

        print(f"[SAM2-Tiny] Propagating masks across all {len(video_frames)} frames...")
        print(f"[SAM2-Tiny] Using temporal consistency for smooth tracking...")

        # Propagate masks to all frames with temporal consistency
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Extract masks for our object
        h = inference_state["video_height"]
        w = inference_state["video_width"]

        masks = []
        for frame_idx in sorted(video_segments.keys()):
            if object_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][object_id][0]  # [H, W]
                masks.append(mask)
            else:
                # Object not detected in this frame, add empty mask
                masks.append(np.zeros((h, w), dtype=bool))

        masks = np.array(masks)  # [T, H, W]

        print(f"[SAM2-Tiny] Tracking complete!")
        print(f"[SAM2-Tiny] Generated masks: {masks.shape}")
        print(f"[SAM2-Tiny] Mask coverage: {masks.sum() / masks.size * 100:.2f}% of pixels")

        return masks


def manual_select_point(frame, tracker=None, video_path=None):
    """
    Manual point selection using mouse click with live mask preview

    Args:
        frame: numpy array [H, W, 3] (RGB)
        tracker: SAM2Tracker instance (optional, for preview)
        video_path: Video path (optional, for preview)

    Returns:
        point: [x, y] or None if cancelled
    """
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    frame_original = frame_bgr.copy()

    selected_point = [None, None]
    preview_mask = None

    def show_mask_preview(point):
        """Generate and show mask preview for clicked point"""
        if tracker is None or video_path is None:
            return None

        print(f"[SAM2-Tiny] Generating mask preview for point {point}...")

        # Initialize SAM2 for single frame
        inference_state = tracker.predictor.init_state(video_path=video_path)

        # Add point prompt
        _, out_obj_ids, out_mask_logits = tracker.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=np.array([point], dtype=np.float32),
            labels=np.array([1], dtype=np.int32)
        )

        # Get mask for first frame only
        mask = (out_mask_logits[0][0] > 0.0).cpu().numpy()

        print(f"[SAM2-Tiny] Preview mask coverage: {mask.sum() / mask.size * 100:.2f}%")
        return mask

    def mouse_callback(event, x, y, flags, param):
        nonlocal preview_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point[0] = x
            selected_point[1] = y

            # Generate mask preview
            preview_mask = show_mask_preview([x, y])

            # Show preview with mask overlay
            display = frame_original.copy()

            if preview_mask is not None:
                # Create green overlay for mask
                overlay = display.copy()
                overlay[preview_mask] = [0, 255, 0]  # Green
                display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

            # Draw circle at clicked point
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

            # Add text instructions
            cv2.putText(display, "SPACE/ENTER = Confirm | ESC = Cancel | Click again = New point",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("SAM2-Tiny: Click on Watermark", display)

    print("\n[SAM2-Tiny] Manual point selection mode:")
    print("  - Click on the watermark object")
    print("  - You'll see a GREEN PREVIEW of the mask")
    print("  - Press SPACE or ENTER to confirm")
    print("  - Press ESC to cancel")
    print("  - Click again to try a different point")

    cv2.namedWindow("SAM2-Tiny: Click on Watermark")
    cv2.setMouseCallback("SAM2-Tiny: Click on Watermark", mouse_callback)
    cv2.imshow("SAM2-Tiny: Click on Watermark", frame_bgr)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("[SAM2-Tiny] Selection cancelled")
            return None
        elif key in [13, 32]:  # ENTER or SPACE
            if selected_point[0] is not None:
                cv2.destroyAllWindows()
                print(f"[SAM2-Tiny] Confirmed point: {selected_point}")
                return selected_point

    cv2.destroyAllWindows()
    return None


def manual_select_bbox(frame):
    """
    Manual bounding box selection using OpenCV window

    Args:
        frame: numpy array [H, W, 3] (RGB)

    Returns:
        bbox: [x1, y1, x2, y2] or None if cancelled
    """
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    print("\n[SAM2-Tiny] Manual selection mode:")
    print("  - Draw a bounding box around the watermark")
    print("  - Press SPACE or ENTER to confirm")
    print("  - Press ESC or C to cancel")

    # Use OpenCV selectROI
    bbox = cv2.selectROI("SAM2-Tiny: Select Watermark", frame_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if bbox[2] == 0 or bbox[3] == 0:
        print("[SAM2-Tiny] Selection cancelled")
        return None

    # Convert from (x, y, w, h) to (x1, y1, x2, y2)
    x, y, w, h = bbox
    bbox_xyxy = [x, y, x + w, y + h]

    print(f"[SAM2-Tiny] Selected bbox: {bbox_xyxy}")

    return bbox_xyxy


if __name__ == "__main__":
    # Example usage
    print("SAM2-Tiny Video Tracker - Example")
    print("="*60)

    # Initialize tracker
    tracker = SAM2Tracker(device="cuda" if torch.cuda.is_available() else "cpu")

    # Example: Load a video and track watermark
    # video_path = "test_video.mp4"
    # cap = cv2.VideoCapture(video_path)
    # frames = []
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frames.append(frame_rgb)
    # cap.release()

    # # Manual selection on first frame
    # bbox = manual_select_bbox(frames[0])
    # if bbox is not None:
    #     masks = tracker.track_from_box(frames, bbox, frame_idx=0)
    #     print(f"Generated {len(masks)} masks!")

    print("\nSAM2-Tiny tracker ready for use!")
