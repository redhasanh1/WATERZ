#!/usr/bin/env python3
"""
SAM2 Tracking Script for WSL2 - Called from Windows Celery worker
Usage:
  Point mode:  python sam2_track_wsl2.py <video_path> --point x,y <output_masks_dir>
  Bbox mode:   python sam2_track_wsl2.py <video_path> --bbox x1,y1,x2,y2 <output_masks_dir>

Uses HYBRID approach:
- Frame 0: TensorRT FP16 (instant)
- Frames 1-N: PyTorch torch.compile() + BFloat16
"""

import os
import sys
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
import argparse
import subprocess

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
SAM2_PATH = Path(__file__).parent / "segment-anything-2"
sys.path.insert(0, str(SAM2_PATH))

# Import SAM2 PyTorch
from sam2.build_sam import build_sam2_video_predictor

# Try to import TensorRT predictor (optional for frame 0)
try:
    from sam2_trt_predictor import SAM2TensorRTPredictor
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("[SAM2-WSL2] TensorRT predictor not available, using PyTorch for all frames")

# Configuration
SAM2_CHECKPOINT = SAM2_PATH / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# TensorRT engines - WSL2 paths
ENCODER_ENGINE = "/mnt/d/watermarkz/sam2_trt_inference/engines_wsl2/sam2_encoder_fp16.engine"
DECODER_ENGINE = "/mnt/d/watermarkz/sam2_trt_inference/engines_wsl2/sam2_decoder_fp16_dynamic.engine"


def enable_optimizations():
    """Enable PyTorch optimizations for WSL2"""
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_properties(0).major

        # TF32 on Ampere+
        if compute_capability >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Memory allocator
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Inductor - DISABLED for faster testing
        # torch._inductor.config.coordinate_descent_tuning = True
        # torch._inductor.config.triton.unique_kernel_names = True
        # torch._inductor.config.fx_graph_cache = True

        # SDPA
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Disable gradients
        torch.set_grad_enabled(False)

        # Async CUDA
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def extract_frames(video_path, output_dir):
    """Extract frames from video"""
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_dir}/{frame_idx:05d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_idx += 1

    cap.release()
    print(f"[SAM2] Extracted {frame_idx} frames @ {fps:.1f} fps")
    return frame_idx, fps


def release_old_frames_complete(inference_state, keep_last=64):
    """
    Complete Det-SAM2 memory management implementation
    Based on Section 3.7 of Det-SAM2 paper
    """
    try:
        if "images" not in inference_state or len(inference_state["images"]) <= keep_last:
            return

        current_count = len(inference_state["images"])
        drop_count = current_count - keep_last

        # Create synchronized arrays with proper index mapping
        new_images = inference_state["images"][drop_count:]
        new_images_idx = inference_state["images_idx"][drop_count:]

        # Update state with synchronized arrays
        inference_state["images"] = new_images
        inference_state["images_idx"] = new_images_idx

        # CRITICAL: Clean output_dict - this is the main memory leak source
        if "output_dict" in inference_state:
            output_dict = inference_state["output_dict"]
            first_kept = new_images_idx[0] if len(new_images_idx) > 0 else 0

            # Clean all frame outputs for dropped frames
            for key in ["non_cond_frame_outputs", "cond_frame_outputs"]:
                if key in output_dict:
                    for old_idx in list(output_dict[key].keys()):
                        if old_idx < first_kept:
                            output_dict[key].pop(old_idx, None)

            # Clean consolidated frame indices
            if "consolidated_frame_inds" in output_dict:
                consolidated = output_dict["consolidated_frame_inds"]
                if "cond_frame_outputs" in consolidated:
                    consolidated["cond_frame_outputs"] = {
                        idx for idx in consolidated["cond_frame_outputs"]
                        if idx >= first_kept
                    }

        # Clean per-object output dictionaries
        if "output_dict_per_obj" in inference_state:
            for obj_id, obj_dict in inference_state["output_dict_per_obj"].items():
                first_kept = new_images_idx[0] if len(new_images_idx) > 0 else 0

                for key in ["non_cond_frame_outputs", "cond_frame_outputs"]:
                    if key in obj_dict:
                        for old_idx in list(obj_dict[key].keys()):
                            if old_idx < first_kept:
                                obj_dict[key].pop(old_idx, None)

        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[SAM2] Det-SAM2 cleanup warning: {e}")


def _track_batch(frames_dir, batch_start, batch_end, output_masks_dir, points, labels, bbox, frame_idx_start):
    """
    Process a single batch of frames - Det-SAM2 batch processing approach
    """
    import shutil
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = batch_end - batch_start

    # Create temporary batch directory with only this batch's frames
    batch_frames_dir = f"/tmp/sam2_batch_{batch_start}"
    if os.path.exists(batch_frames_dir):
        shutil.rmtree(batch_frames_dir)
    os.makedirs(batch_frames_dir, exist_ok=True)

    # Copy only frames for this batch (renumbered from 0)
    print(f"[SAM2-Batch] Copying frames {batch_start}-{batch_end} to batch directory...")
    for i in range(batch_start, batch_end):
        src = os.path.join(frames_dir, f"{i:05d}.jpg")
        dst = os.path.join(batch_frames_dir, f"{i - batch_start:05d}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Load predictor
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
        vos_optimized=False
    )
    torch.cuda.empty_cache()

    # Initialize state for this batch only (much faster!)
    inference_state = predictor.init_state(
        video_path=batch_frames_dir,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
        async_loading_frames=False,
    )

    # Adjust prompt frame index for batch (first batch uses original, others use frame 0 with last mask)
    batch_prompt_frame = 0 if batch_start > 0 else frame_idx_start

    # Add prompt
    if bbox is not None:
        box_array = np.array(bbox, dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=batch_prompt_frame,
            obj_id=1,
            box=box_array
        )
    elif points is not None and len(points) > 0:
        points_array = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int32) if labels else np.ones(len(points), dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=batch_prompt_frame,
            obj_id=1,
            points=points_array,
            labels=labels_array
        )

    # Propagate this batch
    masks_saved = 0
    last_mask = None

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            for frame_idx, obj_ids, mask_logits in tqdm(
                predictor.propagate_in_video(inference_state),
                total=batch_size,
                desc=f"Batch {batch_start//200}"
            ):
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Filter noise
                if np.sum(mask_uint8 > 127) < 100:
                    mask_uint8[:] = 0

                # Save with original frame numbering
                original_frame_idx = batch_start + frame_idx
                cv2.imwrite(f"{output_masks_dir}/{original_frame_idx:05d}.png", mask_uint8)
                masks_saved += 1
                last_mask = mask_uint8

                # Det-SAM2 cleanup
                if masks_saved % 50 == 0:
                    release_old_frames_complete(inference_state, keep_last=64)
                    gc.collect()
                    torch.cuda.empty_cache()

    # Full cleanup between batches
    del predictor
    del inference_state
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Remove batch directory
    shutil.rmtree(batch_frames_dir, ignore_errors=True)

    return masks_saved, last_mask


def track_video_pytorch_only(frames_dir, total_frames, output_masks_dir, points=None, labels=None, bbox=None, frame_idx_start=0):
    """
    Pure PyTorch SAM2 tracking with Det-SAM2 batch processing for long videos.
    Processes videos in batches of 200 frames to prevent memory accumulation during loading.

    Args:
        points: List of (x, y) tuples for multiple click points
        labels: List of labels (1=foreground, 0=background) for each point
    """
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_optimizations()

    # Create output directory
    os.makedirs(output_masks_dir, exist_ok=True)

    BATCH_SIZE = 200  # Process in batches of 200 frames to prevent memory accumulation

    if total_frames <= BATCH_SIZE:
        # For small videos, process normally (single batch)
        print(f"[SAM2-PyTorch] Processing {total_frames} frames in single batch...")
        masks_saved, _ = _track_batch(
            frames_dir, 0, total_frames, output_masks_dir,
            points, labels, bbox, frame_idx_start
        )
    else:
        # For large videos, process in batches
        num_batches = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"[SAM2-PyTorch] Processing {total_frames} frames in {num_batches} batches of {BATCH_SIZE}")

        total_masks_saved = 0
        last_mask = None

        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, total_frames)

            print(f"\n[SAM2-Batch] Processing batch {batch_idx + 1}/{num_batches}: frames {batch_start}-{batch_end}")

            masks_saved, last_mask = _track_batch(
                frames_dir, batch_start, batch_end, output_masks_dir,
                points, labels, bbox, frame_idx_start
            )
            total_masks_saved += masks_saved

            # Force cleanup between batches
            gc.collect()
            torch.cuda.empty_cache()

            print(f"[SAM2-Batch] Batch {batch_idx + 1} complete: {masks_saved} masks saved")

        masks_saved = total_masks_saved

    print(f"[SAM2] Complete! Saved {masks_saved} masks to {output_masks_dir}")
    print(f"[SAM2] VRAM cleanup complete")

    return masks_saved


def convert_windows_to_wsl_path(path):
    """Convert Windows path to WSL2 path if needed"""
    if not path:
        return path
    # Handle D: or d: drive
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        return f'/mnt/{drive}' + path[2:].replace('\\', '/')
    elif '\\' in path:
        return path.replace('\\', '/')
    return path


def main():
    parser = argparse.ArgumentParser(description='SAM2 Video Tracking for WSL2')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('output_masks_dir', help='Output directory for masks')

    # Prompt options - mutually exclusive
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--point', help='Point coordinates as x,y')
    prompt_group.add_argument('--bbox', help='Bounding box as x1,y1,x2,y2')

    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index to start tracking from')
    parser.add_argument('--temp-frames-dir', default='/tmp/sam2_frames', help='Temp dir for frames')

    args = parser.parse_args()

    # Parse prompt
    point = None
    bbox = None

    if args.point:
        try:
            x, y = map(int, args.point.split(','))
            point = (x, y)
            print(f"[SAM2] Using point prompt: {point}")
        except:
            print(f"[ERROR] Invalid point format: {args.point}. Use x,y (e.g., 200,150)")
            sys.exit(1)
    elif args.bbox:
        try:
            coords = list(map(int, args.bbox.split(',')))
            if len(coords) != 4:
                raise ValueError("Need exactly 4 coordinates")
            bbox = coords  # [x1, y1, x2, y2]
            print(f"[SAM2] Using bbox prompt: {bbox}")
        except:
            print(f"[ERROR] Invalid bbox format: {args.bbox}. Use x1,y1,x2,y2 (e.g., 100,100,300,300)")
            sys.exit(1)

    # Convert Windows paths to WSL2 paths
    video_path = convert_windows_to_wsl_path(args.video_path)
    output_masks_dir = convert_windows_to_wsl_path(args.output_masks_dir)
    temp_frames_dir = convert_windows_to_wsl_path(args.temp_frames_dir)

    print(f"[SAM2] Video: {video_path}")
    print(f"[SAM2] Output: {output_masks_dir}")
    print(f"[SAM2] Frame idx: {args.frame_idx}")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Extract frames
    total_frames, fps = extract_frames(video_path, temp_frames_dir)

    # Run tracking (pure PyTorch - simpler and still fast)
    masks_saved = track_video_pytorch_only(
        temp_frames_dir,
        total_frames,
        output_masks_dir,
        point=point,
        bbox=bbox,
        frame_idx_start=args.frame_idx
    )

    # Output JSON for Windows to read
    result = {
        'status': 'success',
        'total_frames': total_frames,
        'masks_saved': masks_saved,
        'fps': fps,
        'masks_dir': output_masks_dir
    }
    print(f"\n[RESULT] {json.dumps(result)}")


if __name__ == '__main__':
    main()
