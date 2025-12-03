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

        # Inductor
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

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


def track_video_pytorch_only(frames_dir, total_frames, output_masks_dir, point=None, bbox=None, frame_idx_start=0):
    """
    Pure PyTorch SAM2 tracking - works for both point and bbox prompts
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_optimizations()

    # Create output directory
    os.makedirs(output_masks_dir, exist_ok=True)

    print(f"[SAM2-PyTorch] Loading SAM2 video predictor...")

    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
        vos_optimized=True
    )

    torch.cuda.empty_cache()

    # Initialize state
    inference_state = predictor.init_state(video_path=frames_dir)

    # Add prompt (point or bbox) to initialize tracking
    if bbox is not None:
        # bbox format: [x1, y1, x2, y2]
        box_array = np.array(bbox, dtype=np.float32)
        print(f"[SAM2-PyTorch] Adding bbox prompt: {bbox} on frame {frame_idx_start}")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx_start,
            obj_id=1,
            box=box_array
        )
    elif point is not None:
        # point format: (x, y)
        points_array = np.array([[point[0], point[1]]], dtype=np.float32)
        labels_array = np.array([1], dtype=np.int32)  # 1 = foreground
        print(f"[SAM2-PyTorch] Adding point prompt: {point} on frame {frame_idx_start}")
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx_start,
            obj_id=1,
            points=points_array,
            labels=labels_array
        )
    else:
        raise ValueError("Must provide either point or bbox")

    print(f"[SAM2-PyTorch] Propagating {total_frames} frames...")

    # Propagate
    masks_saved = 0
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            for frame_idx, obj_ids, mask_logits in tqdm(
                predictor.propagate_in_video(inference_state),
                total=total_frames,
                desc="SAM2 Tracking"
            ):
                # Get mask
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Detect empty mask (object off-screen or lost)
                if np.sum(mask_uint8) == 0:
                    print(f"[SAM2] Frame {frame_idx}: Object not visible (empty mask)")

                # Save mask
                cv2.imwrite(f"{output_masks_dir}/{frame_idx:05d}.png", mask_uint8)
                masks_saved += 1

    print(f"[SAM2] Complete! Saved {masks_saved} masks to {output_masks_dir}")
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
