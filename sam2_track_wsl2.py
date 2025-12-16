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
    Memory cleanup for SAM2 - SAFE version.

    IMPORTANT: Cannot truncate images tensor during propagation because
    propagate_in_video generator uses original frame indices.
    Only clean output_dict entries for memory savings.
    """
    try:
        # DON'T touch images tensor - generator uses original indices!
        # Only clean non_cond_frame_outputs to save some memory

        if "output_dict" in inference_state:
            output_dict = inference_state["output_dict"]

            # Only clean non-conditioning frame outputs (keep last N)
            if "non_cond_frame_outputs" in output_dict:
                keys = sorted(output_dict["non_cond_frame_outputs"].keys())
                if len(keys) > keep_last:
                    for old_idx in keys[:-keep_last]:
                        output_dict["non_cond_frame_outputs"].pop(old_idx, None)

            # DO NOT touch cond_frame_outputs - SAM2 needs them

        # Clean per-object output dictionaries
        if "output_dict_per_obj" in inference_state:
            for obj_id, obj_dict in inference_state["output_dict_per_obj"].items():
                if "non_cond_frame_outputs" in obj_dict:
                    keys = sorted(obj_dict["non_cond_frame_outputs"].keys())
                    if len(keys) > keep_last:
                        for old_idx in keys[:-keep_last]:
                            obj_dict["non_cond_frame_outputs"].pop(old_idx, None)

        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[SAM2-Memory] Cleanup warning: {e}")


def load_frames_as_tensors(frames_dir, start_idx, end_idx, img_mean, img_std, image_size):
    """
    Load frames from disk and convert to tensors matching SAM2's format.

    Returns list of tensors ready to extend inference_state["images"]
    """
    from PIL import Image
    import torchvision.transforms as transforms

    # SAM2 uses this transform internally
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    images = []
    for i in range(start_idx, end_idx):
        frame_path = os.path.join(frames_dir, f"{i:05d}.jpg")
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)

    return images


def track_video_pytorch_only(frames_dir, total_frames, output_masks_dir, points=None, labels=None, bbox=None, frame_idx_start=0):
    """
    SAM2 video tracking using OFFICIAL memory management parameters.

    Uses start_frame_idx and max_frame_num_to_track for chunked processing
    instead of manual tensor manipulation.

    Args:
        points: List of (x, y) tuples for multiple click points
        labels: List of labels (1=foreground, 0=background) for each point
    """
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_optimizations()

    # Create output directory
    os.makedirs(output_masks_dir, exist_ok=True)

    CHUNK_SIZE = 128  # Process 128 frames at a time (memory optimized for long videos)

    # Build predictor
    print(f"[SAM2-Official] Building predictor with official memory management...")
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
        vos_optimized=False
    )
    torch.cuda.empty_cache()

    try:
        # Initialize state - KEEP IN VRAM for speed (user has 24GB 4090)
        print(f"[SAM2-Official] Initializing state for {total_frames} frames (VRAM mode)...")
        inference_state = predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=False,   # KEEP IN VRAM!
            offload_state_to_cpu=False,   # KEEP IN VRAM!
            async_loading_frames=True,    # Enable async for speed
        )

        # Add initial prompt
        if bbox is not None:
            box_array = np.array(bbox, dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx_start,
                obj_id=1,
                box=box_array
            )
            print(f"[SAM2-Official] Added bbox prompt at frame {frame_idx_start}")
        elif points is not None and len(points) > 0:
            points_array = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32) if labels else np.ones(len(points), dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx_start,
                obj_id=1,
                points=points_array,
                labels=labels_array
            )
            print(f"[SAM2-Official] Added point prompt at frame {frame_idx_start}")

        masks_saved = 0
        num_chunks = (total_frames + CHUNK_SIZE - 1) // CHUNK_SIZE

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):

                # Process in chunks using SAM2's official parameters
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * CHUNK_SIZE
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)

                    print(f"\n[SAM2-Chunk] Processing chunk {chunk_idx + 1}/{num_chunks}: frames {chunk_start}-{chunk_end}")

                    chunk_masks = 0
                    for frame_idx, obj_ids, mask_logits in tqdm(
                        predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=chunk_start,
                            max_frame_num_to_track=CHUNK_SIZE
                        ),
                        total=chunk_end - chunk_start,
                        desc=f"Chunk {chunk_idx + 1}"
                    ):
                        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                        mask_uint8 = (mask * 255).astype(np.uint8)

                        # Filter noise
                        if np.sum(mask_uint8 > 127) < 100:
                            mask_uint8[:] = 0

                        cv2.imwrite(f"{output_masks_dir}/{frame_idx:05d}.png", mask_uint8)
                        masks_saved += 1
                        chunk_masks += 1

                        # CRITICAL: Release memory immediately to prevent accumulation
                        del mask_logits, mask

                    print(f"[SAM2-Chunk] Chunk {chunk_idx + 1} complete: {chunk_masks} masks")

                    # Memory cleanup between chunks
                    gc.collect()
                    torch.cuda.empty_cache()

        print(f"[SAM2] Complete! Saved {masks_saved} masks to {output_masks_dir}")
        return masks_saved

    finally:
        # Always cleanup predictor
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
    # Convert single point to list format expected by function
    points_list = [point] if point else None
    labels_list = [1] if point else None  # 1 = foreground

    masks_saved = track_video_pytorch_only(
        temp_frames_dir,
        total_frames,
        output_masks_dir,
        points=points_list,
        labels=labels_list,
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
