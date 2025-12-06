"""
WSL2 Celery worker for SAM2 mask generation (FULL FPS).

Runs entirely inside WSL2, connected to the same Redis broker as Windows.
Generates masks into a shared path (e.g., /mnt/d/watermarkz/temp/<video_id>_sam2_masks),
so the Windows Celery task can immediately consume them without extra copies.
"""

import os
import sys
import json
import glob
import time
import zipfile
import shutil
from celery import Celery

# Resolve base dir assuming this file resides in repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Redis URL (shared with Windows worker)
REDIS_URL = os.getenv('REDIS_URL', 'redis://:watermarkz_secure_2024@localhost:6379/0')
redis_path = os.path.join(BASE_DIR, 'redis_url.txt')
if os.path.exists(redis_path):
    with open(redis_path, 'r') as f:
        REDIS_URL = f.read().strip()

celery = Celery('wsl_sam2_worker', broker=REDIS_URL, backend=REDIS_URL)
celery.conf.task_track_started = True
celery.conf.broker_connection_retry_on_startup = True

# Add local path for imports
sys.path.insert(0, BASE_DIR)


@celery.task(name='sam2.generate_masks_fullfps', bind=True)
def generate_masks_fullfps(self, video_path, masks_dir, prompt_mode='point', points=None, labels=None, bbox=None, frame_idx=0, api_base=None):
    """
    Generate SAM2 masks at full FPS inside WSL2.

    Args:
        video_path (str): Windows or WSL path to video (or remote path like /data/uploads/...)
        masks_dir (str): Windows or WSL path for output masks
        prompt_mode (str): 'point' or 'bbox'
        points (list): List of (x, y) tuples if prompt_mode is 'point' (supports multiple clicks!)
        labels (list): List of labels (1=foreground, 0=background) for each point
        bbox (list): [x1, y1, x2, y2] if prompt_mode is 'bbox'
        frame_idx (int): starting frame index
        api_base (str): Base URL for downloading video if path is remote
    Returns:
        dict: {status, masks_dir, masks_saved, total_frames}
    """
    import os
    import cv2
    import numpy as np
    from pathlib import Path
    import requests

    # Import helper module from repo (runs under WSL)
    import sam2_track_wsl2 as sam2w
    import subprocess, shlex, shutil

    def to_wsl_path(p: str) -> str:
        if not isinstance(p, str):
            return p
        # Already looks like WSL
        if p.startswith('/mnt/') or p.startswith('/'):
            return p
        # Convert D:\... → /mnt/d/...
        if len(p) >= 2 and p[1] == ':' and p[0].isalpha():
            drive = p[0].lower()
            rest = p[2:].replace('\\', '/')
            rest = rest.lstrip('/')
            return f"/mnt/{drive}/{rest}"
        return p.replace('\\', '/')

    video_path_wsl = to_wsl_path(video_path)
    masks_dir_wsl = to_wsl_path(masks_dir)
    temp_frames_dir = to_wsl_path('/tmp/sam2_frames')

    # If video doesn't exist locally OR is a Railway path (/data/...), download it from api_base
    is_railway_path = video_path.startswith('/data/') or video_path_wsl.startswith('/data/')
    if (is_railway_path or not os.path.exists(video_path_wsl)) and api_base:
        filename = os.path.basename(video_path)
        download_url = f"{api_base}/uploads/{filename}"
        local_video = f"/tmp/{filename}"
        print(f"[WSL] Downloading video from {download_url}...")
        self.update_state(state='PROCESSING', meta={'status': 'Downloading video', 'progress': 2})
        r = requests.get(download_url, timeout=300)
        r.raise_for_status()
        with open(local_video, 'wb') as f:
            f.write(r.content)
        video_path_wsl = local_video
        print(f"[WSL] Downloaded {len(r.content) / 1024 / 1024:.1f} MB to {local_video}")

    os.makedirs(masks_dir_wsl, exist_ok=True)

    # Optional: track at 10fps for speed, then Windows will expand masks back to full-FPS
    use_10fps_env = os.getenv('SAM2_WSL_TRACK_10FPS', '0').lower() in ('1','true','yes','on')
    # If Windows already created a 10fps video (e.g., *_tracking_10fps.mp4), do NOT try to downsample again
    already_10 = ('_tracking_10fps' in os.path.basename(video_path_wsl)) or video_path_wsl.lower().endswith('_10fps.mp4')
    ffmpeg_ok = shutil.which('ffmpeg') is not None
    track_src = video_path_wsl
    if use_10fps_env and not already_10 and ffmpeg_ok:
        video_10_path = f"/tmp/sam2_{os.path.basename(video_path_wsl)}_10fps.mp4"
        cmd = f"ffmpeg -y -i {shlex.quote(video_path_wsl)} -vf fps=10 -c:v libx264 -preset ultrafast -crf 18 -an {shlex.quote(video_10_path)}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            track_src = video_10_path
        except Exception as e:
            # Fallback: if ffmpeg fails, track at full FPS instead of crashing
            print(f"[WSL] Warning: ffmpeg downsample failed ({e}); tracking at full FPS")

    # Extract frames
    self.update_state(state='PROCESSING', meta={'status': 'Extracting frames', 'progress': 5})
    total_frames, fps = sam2w.extract_frames(track_src, temp_frames_dir)

    # Run tracking (PyTorch-only in-process; TensorRT hybrid can be added later)
    self.update_state(state='PROCESSING', meta={'status': 'Tracking SAM2', 'progress': 20})
    if prompt_mode == 'point' and points is not None and len(points) > 0:
        # Convert points to list of (x, y) tuples
        pts = [(int(p[0]), int(p[1])) for p in points]
        # Labels default to all foreground (1) if not provided
        lbls = labels if labels is not None else [1] * len(pts)
        print(f"[WSL-SAM2] Using {len(pts)} point(s): {pts}")
        masks_saved = sam2w.track_video_pytorch_only(
            temp_frames_dir, total_frames, masks_dir_wsl,
            points=pts, labels=lbls, bbox=None, frame_idx_start=int(frame_idx)
        )
    elif prompt_mode == 'bbox' and bbox is not None:
        bb = [int(x) for x in bbox[:4]]
        masks_saved = sam2w.track_video_pytorch_only(
            temp_frames_dir, total_frames, masks_dir_wsl,
            points=None, labels=None, bbox=bb, frame_idx_start=int(frame_idx)
        )
    else:
        raise ValueError('Invalid prompt: provide points or bbox')

    # Force VRAM cleanup after tracking
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # Upload masks to B2 CDN
    self.update_state(state='PROCESSING', meta={'status': 'Uploading masks to B2', 'progress': 90})
    masks_url = upload_masks_to_b2(masks_dir_wsl, video_path)

    return {
        'status': 'success',
        'masks_url': masks_url,
        'masks_dir': None,  # No local path - use B2 URL
        'masks_saved': int(masks_saved),
        'total_frames': int(total_frames),
        'fps': float(fps),  # tracking fps (10 or full), used for expansion on Windows
    }


def upload_masks_to_b2(masks_dir, video_path):
    """
    Zip and upload masks to B2 CDN, return CDN URL.
    Cleans up local masks after upload.
    """
    from b2sdk.v2 import B2Api, InMemoryAccountInfo

    B2_KEY_ID = os.getenv('B2_KEY_ID')
    B2_APP_KEY = os.getenv('B2_APP_KEY')
    B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
    B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

    if not B2_KEY_ID or not B2_APP_KEY:
        print("[B2] Warning: B2 credentials not set, returning local path")
        return None

    # Extract video_id from path for naming
    video_basename = os.path.basename(video_path)
    video_id = os.path.splitext(video_basename)[0]
    # Remove any suffix like _tracking_10fps
    if '_tracking_' in video_id:
        video_id = video_id.split('_tracking_')[0]

    # Zip masks
    zip_path = f"/tmp/{video_id}_masks.zip"
    mask_files = sorted(glob.glob(f"{masks_dir}/*.png"))
    print(f"[B2] Zipping {len(mask_files)} masks...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mask_file in mask_files:
            zf.write(mask_file, os.path.basename(mask_file))

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[B2] Zip created: {zip_size_mb:.1f} MB")

    # Upload to B2
    print(f"[B2] Uploading to {B2_BUCKET}...")
    upload_start = time.time()

    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
    bucket = b2_api.get_bucket_by_name(B2_BUCKET)

    timestamp = int(time.time())
    remote_path = f"masks/{timestamp}_{video_id}_masks.zip"
    bucket.upload_local_file(local_file=zip_path, file_name=remote_path)

    masks_url = f"{B2_CDN_URL}/{remote_path}"
    upload_time = time.time() - upload_start
    print(f"[B2] Upload complete in {upload_time:.1f}s: {masks_url}")

    # Cleanup local files
    shutil.rmtree(masks_dir, ignore_errors=True)
    os.remove(zip_path)
    print(f"[B2] Local masks cleaned up")

    return masks_url


def to_wsl_path(p: str) -> str:
    """Convert Windows path to WSL path."""
    if not isinstance(p, str):
        return p
    if p.startswith('/mnt/') or p.startswith('/'):
        return p
    if len(p) >= 2 and p[1] == ':' and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].replace('\\', '/')
        rest = rest.lstrip('/')
        return f"/mnt/{drive}/{rest}"
    return p.replace('\\', '/')


@celery.task(name='yolo.generate_masks', bind=True)
def generate_masks_yolo(self, video_path, masks_dir, confidence_threshold=0.3, padding=30, api_base=None):
    """
    Generate masks using YOLO detection (automatic, no user clicks).

    Uses TensorRT engine if available (20-30x faster), falls back to PyTorch.

    Args:
        video_path: Path to video file
        masks_dir: Output directory for masks
        confidence_threshold: YOLO confidence threshold
        padding: Padding around detections
        api_base: Base URL for downloading video if path is remote

    Returns:
        dict with masks_url, total_frames, etc.
    """
    import cv2
    import numpy as np
    import requests

    video_path_wsl = to_wsl_path(video_path)
    masks_dir_wsl = to_wsl_path(masks_dir)
    os.makedirs(masks_dir_wsl, exist_ok=True)

    # If video is a Railway path (/data/...), download it from api_base
    is_railway_path = video_path.startswith('/data/') or video_path_wsl.startswith('/data/')
    if (is_railway_path or not os.path.exists(video_path_wsl)) and api_base:
        filename = os.path.basename(video_path)
        download_url = f"{api_base}/uploads/{filename}"
        local_video = f"/tmp/{filename}"
        print(f"[YOLO] Downloading video from {download_url}...")
        self.update_state(state='PROCESSING', meta={'status': 'Downloading video', 'progress': 2})
        r = requests.get(download_url, timeout=300)
        r.raise_for_status()
        with open(local_video, 'wb') as f:
            f.write(r.content)
        video_path_wsl = local_video
        print(f"[YOLO] Downloaded {len(r.content) / 1024 / 1024:.1f} MB to {local_video}")

    self.update_state(state='PROCESSING', meta={'status': 'Loading YOLO model', 'progress': 5})

    # Try TensorRT engine first (20-30x faster), fall back to PyTorch
    from ultralytics import YOLO

    trt_engine_path = '/mnt/d/watermarkz/runs/detect/new_sora_watermark/weights/best_fp16_batch_wsl.engine'
    pt_model_path = '/mnt/d/watermarkz/runs/detect/new_sora_watermark/weights/best.pt'

    using_tensorrt = False
    if os.path.exists(trt_engine_path):
        try:
            model = YOLO(trt_engine_path, task='detect')
            print(f"[YOLO] Loaded TensorRT engine: {trt_engine_path}")
            print(f"[YOLO] Expected: ~1-2ms/frame (20-30x faster than PyTorch)")
            using_tensorrt = True
        except Exception as e:
            print(f"[YOLO] TensorRT load failed: {e}, falling back to PyTorch")

    if not using_tensorrt:
        if not os.path.exists(pt_model_path):
            raise RuntimeError(f"YOLO model not found: {pt_model_path}")
        model = YOLO(pt_model_path)
        print(f"[YOLO] Loaded PyTorch model: {pt_model_path}")
        print(f"[YOLO] TIP: Run build_yolo_trt_wsl.sh to build TensorRT engine for 20-30x speedup")

    # Open video
    self.update_state(state='PROCESSING', meta={'status': 'Opening video', 'progress': 10})
    cap = cv2.VideoCapture(video_path_wsl)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path_wsl}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[YOLO] Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")

    # Process frames
    masks_saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress every 10 frames
        if frame_idx % 10 == 0:
            progress = 10 + int(80 * frame_idx / total_frames)
            self.update_state(state='PROCESSING', meta={
                'status': f'Detecting frame {frame_idx}/{total_frames}',
                'progress': progress
            })

        # Pad frame to 640x640 for TensorRT engine (same as Windows yolo_detector.py)
        h, w = frame.shape[:2]
        scale = min(640.0 / w, 640.0 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w, pad_h = 640 - new_w, 640 - new_h
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Run YOLO detection on padded 640x640 image
        results = model(padded, conf=confidence_threshold, device='cuda', verbose=False, imgsz=640)

        # Create mask from detections
        mask = np.zeros((height, width), dtype=np.uint8)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Remove padding offset and scale back to original coordinates
                x1 = (x1 - left) / scale
                y1 = (y1 - top) / scale
                x2 = (x2 - left) / scale
                y2 = (y2 - top) / scale

                # Add detection padding and clamp to image bounds
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(width, int(x2) + padding)
                y2 = min(height, int(y2) + padding)

                # Fill mask
                mask[y1:y2, x1:x2] = 255

        # Save mask
        mask_path = os.path.join(masks_dir_wsl, f"{frame_idx:05d}.png")
        cv2.imwrite(mask_path, mask)
        masks_saved += 1
        frame_idx += 1

    cap.release()
    print(f"[YOLO] Generated {masks_saved} masks")

    # Cleanup VRAM
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Upload to B2
    self.update_state(state='PROCESSING', meta={'status': 'Uploading masks to B2', 'progress': 90})
    masks_url = upload_masks_to_b2(masks_dir_wsl, video_path)

    return {
        'status': 'success',
        'masks_url': masks_url,
        'masks_dir': None,
        'masks_saved': int(masks_saved),
        'total_frames': int(total_frames),
        'fps': float(fps),
        'mode': 'yolo',  # Tell receiver this was YOLO mode (use 4 workers)
    }
