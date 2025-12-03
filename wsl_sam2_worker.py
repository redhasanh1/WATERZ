"""
WSL2 Celery worker for SAM2 mask generation (FULL FPS).

Runs entirely inside WSL2, connected to the same Redis broker as Windows.
Generates masks into a shared path (e.g., /mnt/d/watermarkz/temp/<video_id>_sam2_masks),
so the Windows Celery task can immediately consume them without extra copies.
"""

import os
import sys
import json
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
def generate_masks_fullfps(self, video_path, masks_dir, prompt_mode='point', point=None, bbox=None, frame_idx=0):
    """
    Generate SAM2 masks at full FPS inside WSL2.

    Args:
        video_path (str): Windows or WSL path to video
        masks_dir (str): Windows or WSL path for output masks
        prompt_mode (str): 'point' or 'bbox'
        point (tuple/list): (x, y) if prompt_mode is 'point'
        bbox (list): [x1, y1, x2, y2] if prompt_mode is 'bbox'
        frame_idx (int): starting frame index
    Returns:
        dict: {status, masks_dir, masks_saved, total_frames}
    """
    import os
    import cv2
    import numpy as np
    from pathlib import Path

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
    if prompt_mode == 'point' and point is not None:
        pt = (int(point[0]), int(point[1]))
        masks_saved = sam2w.track_video_pytorch_only(
            temp_frames_dir, total_frames, masks_dir_wsl,
            point=pt, bbox=None, frame_idx_start=int(frame_idx)
        )
    elif prompt_mode == 'bbox' and bbox is not None:
        bb = [int(x) for x in bbox[:4]]
        masks_saved = sam2w.track_video_pytorch_only(
            temp_frames_dir, total_frames, masks_dir_wsl,
            point=None, bbox=bb, frame_idx_start=int(frame_idx)
        )
    else:
        raise ValueError('Invalid prompt: provide point or bbox')

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

    return {
        'status': 'success',
        'masks_dir': masks_dir,
        'masks_saved': int(masks_saved),
        'total_frames': int(total_frames),
        'fps': float(fps),  # tracking fps (10 or full), used for expansion on Windows
    }
