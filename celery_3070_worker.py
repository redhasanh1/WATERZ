"""
RTX 3070 Processing Worker - Celery Worker for Segment Processing

This runs on local 3070s and handles:
1. Pulling segment jobs from queue
2. Downloading frames/masks from 4090 server
3. Running ProPainter inpainting
4. Uploading results back

Architecture:
- 4090 (cloud): Detection + segment job creation
- 3070s (this): Process segment jobs from queue (ProPainter inpainting)
"""

import os
import sys
import json
import time
import shutil
import subprocess
import zipfile
import requests
import numpy as np
from pathlib import Path

# Add python_packages to path
sys.path.insert(0, 'python_packages')
# Add faster-propainter-main for local testing (watermark module)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faster-propainter-main'))

# ============================================================================
# GPU OPTIMIZATIONS FOR RTX 3070
# ============================================================================

# Disable Flash Attention (not supported on RTX 3070)
os.environ['ENABLE_FLASH_ATTENTION'] = '0'
os.environ['TORCH_CUDAGRAPHS'] = '0'
os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'

# Disable torch.compile everywhere - Triton fails on Salad runtime images
os.environ['USE_TORCH_COMPILE_RAFT'] = '0'
os.environ['USE_TORCH_COMPILE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import cv2
from celery import Celery
from celery.signals import worker_process_init
from b2sdk.v2 import B2Api, InMemoryAccountInfo

# Import ProPainter
from watermark import pipeline as faster_propainter_pipeline

# Import crop utilities for 20x speedup
from crop_utils import calculate_crop_region

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Redis configuration - load from redis_url.txt or env
REDIS_URL = None
redis_url_file = os.path.join(SCRIPT_DIR, 'redis_url.txt')
if os.path.exists(redis_url_file):
    with open(redis_url_file, 'r') as f:
        REDIS_URL = f.read().strip()
    print(f"[OK] Loaded Redis URL from {redis_url_file}")

broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL") or REDIS_URL
backend = os.getenv("CELERY_RESULT_BACKEND") or broker

if not broker:
    raise ValueError("Redis URL is required - set CELERY_BROKER_URL or REDIS_URL")

print(f"[3070 WORKER] Celery Broker = {broker}")

# Directory configuration
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Railway URL for uploading results
RAILWAY_URL = None
tunnel_url_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
if os.path.exists(tunnel_url_file):
    with open(tunnel_url_file, 'r') as f:
        RAILWAY_URL = f.read().strip()
    print(f"[OK] Loaded Railway URL: {RAILWAY_URL}")

# ============================================================================
# B2 + CLOUDFLARE CONFIGURATION
# ============================================================================

B2_KEY_ID = '00539db5c1104b50000000002'
B2_APPLICATION_KEY = 'K005HJKUP7ahSNJ1wgQHDDJ+uEATiU4'
B2_BUCKET_NAME = 'watermarkz'
CLOUDFLARE_WORKER_URL = 'https://markz.humblewoslayer.workers.dev'

# B2 API (initialized lazily)
_b2_api = None
_b2_bucket = None

def get_b2_bucket():
    """Get B2 bucket (lazy initialization)"""
    global _b2_api, _b2_bucket
    if _b2_bucket is None:
        print("[B2] Initializing B2 API...")
        info = InMemoryAccountInfo()
        _b2_api = B2Api(info)
        _b2_api.authorize_account("production", B2_KEY_ID, B2_APPLICATION_KEY)
        _b2_bucket = _b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        print(f"[B2] Connected to bucket: {B2_BUCKET_NAME}")
    return _b2_bucket

# ============================================================================
# CELERY SETUP
# ============================================================================

celery = Celery('worker_3070', broker=broker, backend=backend)

celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    worker_prefetch_multiplier=1,  # Only 1 segment at a time per worker
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_dynamic_subvideo_length(width, height):
    """Calculate optimal subvideo length based on resolution"""
    pixels = width * height
    if pixels <= 480 * 360:
        return 100
    elif pixels <= 854 * 480:
        return 80
    elif pixels <= 1280 * 720:
        return 60
    else:
        return 50

def download_file(url, dest_path):
    """Download file from URL"""
    print(f"[3070] Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[3070] Downloaded: {dest_path}")

def upload_result_to_railway(video_id, segment_index, output_frames_dir):
    """
    Zip output frames and upload to Railway.
    Returns result URL.
    """
    if not RAILWAY_URL:
        print("[3070] WARNING: No Railway URL, skipping upload")
        return None

    # Create zip of output frames
    zip_filename = f"{video_id}_seg{segment_index}_result.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    print(f"[3070] Creating result zip: {zip_filename}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for frame_file in sorted(os.listdir(output_frames_dir)):
            if frame_file.endswith('.png'):
                frame_path = os.path.join(output_frames_dir, frame_file)
                zf.write(frame_path, frame_file)

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[3070] Result zip: {zip_size:.2f} MB")

    # Upload to Railway
    upload_url = f"{RAILWAY_URL}/api/upload-result"
    print(f"[3070] Uploading to Railway: {upload_url}")

    start_time = time.time()
    with open(zip_path, 'rb') as f:
        files = {'file': (zip_filename, f, 'application/zip')}
        data = {'filename': zip_filename}
        response = requests.post(upload_url, files=files, data=data, timeout=300)

    upload_time = time.time() - start_time
    speed = zip_size / upload_time if upload_time > 0 else 0
    print(f"[3070] Uploaded in {upload_time:.2f}s ({speed:.2f} MB/s)")

    # Clean up local zip
    os.remove(zip_path)

    if response.ok:
        result = response.json()
        result_url = result.get('result_url')
        print(f"[3070] Result URL: {result_url}")
        return result_url
    else:
        print(f"[3070] Upload failed: {response.status_code} - {response.text}")
        return None


def upload_result_to_b2(video_id, segment_index, output_frames_dir):
    """
    Zip output frames and upload to B2.
    Returns Cloudflare URL for download.
    """
    # Create zip of output frames
    zip_filename = f"{video_id}_seg{segment_index}_result.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    print(f"[B2] Creating result zip: {zip_filename}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for frame_file in sorted(os.listdir(output_frames_dir)):
            if frame_file.endswith('.png'):
                frame_path = os.path.join(output_frames_dir, frame_file)
                zf.write(frame_path, frame_file)

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[B2] Result zip: {zip_size:.2f} MB")

    # Upload to B2
    bucket = get_b2_bucket()
    b2_path = f"results/{video_id}/{zip_filename}"

    print(f"[B2] Uploading to B2: {b2_path}...")
    start_time = time.time()

    bucket.upload_local_file(
        local_file=zip_path,
        file_name=b2_path
    )

    upload_time = time.time() - start_time
    speed = zip_size / upload_time if upload_time > 0 else 0
    print(f"[B2] Uploaded in {upload_time:.2f}s ({speed:.2f} MB/s)")

    # Clean up local zip
    os.remove(zip_path)

    # Return Cloudflare URL
    cloudflare_url = f"{CLOUDFLARE_WORKER_URL}/{b2_path}"
    print(f"[B2] Result available at: {cloudflare_url}")
    return cloudflare_url


def upload_final_video_to_b2(video_id, video_path):
    """
    Upload assembled final video to B2 and return Cloudflare URL.
    """
    if not os.path.exists(video_path):
        print(f"[B2] ERROR: Video file not found: {video_path}")
        return None

    file_size = os.path.getsize(video_path) / (1024 * 1024)
    print(f"[B2] Uploading final video: {os.path.basename(video_path)} ({file_size:.2f} MB)")

    bucket = get_b2_bucket()
    b2_path = f"final/{video_id}/{os.path.basename(video_path)}"

    start_time = time.time()
    bucket.upload_local_file(
        local_file=video_path,
        file_name=b2_path
    )

    upload_time = time.time() - start_time
    speed = file_size / upload_time if upload_time > 0 else 0
    print(f"[B2] Uploaded final video in {upload_time:.2f}s ({speed:.2f} MB/s)")

    cloudflare_url = f"{CLOUDFLARE_WORKER_URL}/{b2_path}"
    print(f"[B2] Final video available at: {cloudflare_url}")
    return cloudflare_url


def download_and_extract_from_cloudflare(cloudflare_url, work_dir):
    """
    Download zip from Cloudflare and extract frames+masks.
    Returns (frames_dir, masks_dir)
    """
    # Download zip file
    zip_path = os.path.join(work_dir, "segment.zip")

    print(f"[3070] Downloading from Cloudflare: {cloudflare_url}")
    start_time = time.time()

    response = requests.get(cloudflare_url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = 0
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)

    download_time = time.time() - start_time
    size_mb = total_size / (1024 * 1024)
    speed = size_mb / download_time if download_time > 0 else 0
    print(f"[3070] Downloaded {size_mb:.2f} MB in {download_time:.2f}s ({speed:.2f} MB/s)")

    # Extract zip
    frames_dir = os.path.join(work_dir, 'frames')
    masks_dir = os.path.join(work_dir, 'masks')

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    print(f"[3070] Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.startswith('frames/'):
                # Extract frame to frames_dir
                data = zf.read(name)
                filename = os.path.basename(name)
                if filename:
                    with open(os.path.join(frames_dir, filename), 'wb') as f:
                        f.write(data)
            elif name.startswith('masks/'):
                # Extract mask to masks_dir
                data = zf.read(name)
                filename = os.path.basename(name)
                if filename:
                    with open(os.path.join(masks_dir, filename), 'wb') as f:
                        f.write(data)

    # Clean up zip
    os.remove(zip_path)

    num_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    num_masks = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    print(f"[3070] Extracted {num_frames} frames and {num_masks} masks")

    return frames_dir, masks_dir


def download_video_and_extract_frames(video_url, work_dir, start_frame, end_frame):
    """
    Download video from URL and extract frames for segment range.
    Returns frames_dir path.
    """
    video_path = os.path.join(work_dir, "video.mp4")
    frames_dir = os.path.join(work_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    # Download video
    print(f"[3070] Downloading video: {video_url}")
    start_time = time.time()

    response = requests.get(video_url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = 0
    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)

    download_time = time.time() - start_time
    size_mb = total_size / (1024 * 1024)
    speed = size_mb / download_time if download_time > 0 else 0
    print(f"[3070] Downloaded video: {size_mb:.2f} MB in {download_time:.2f}s ({speed:.2f} MB/s)")

    # Extract frames for this segment range using ffmpeg NVDEC (GPU-accelerated!)
    print(f"[3070] Extracting frames {start_frame}-{end_frame} using NVDEC...")
    extract_start = time.time()

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-hwaccel_output_format', 'cuda',
        '-i', video_path,
        '-vf', f"select='between(n\\,{start_frame}\\,{end_frame})',hwdownload,format=bgr24",
        '-vsync', '0',
        '-start_number', '0',
        os.path.join(frames_dir, '%04d.png')
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg NVDEC failed: {result.stderr}")

    frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    extract_time = time.time() - extract_start
    fps = frame_count / extract_time if extract_time > 0 else 0
    print(f"[3070] NVDEC extracted {frame_count} frames in {extract_time:.2f}s ({fps:.0f} fps)")

    # Clean up video file to save space
    os.remove(video_path)

    print(f"[3070] Extracted {frame_count} frames")
    return frames_dir


# Global cache for all masks (download once per video, reuse for all segments)
_all_masks_cache = {}  # video_id -> masks_dir path


def download_all_masks_cached(video_id, all_masks_url):
    """
    Download ALL masks once and cache for all segments.
    Returns masks_dir path (cached on subsequent calls).
    """
    global _all_masks_cache

    # Check cache first
    if video_id in _all_masks_cache:
        cached_dir = _all_masks_cache[video_id]
        if os.path.exists(cached_dir):
            num_masks = len([f for f in os.listdir(cached_dir) if f.endswith('.png')])
            print(f"[3070] Using CACHED masks for {video_id} ({num_masks} masks)")
            return cached_dir
        else:
            del _all_masks_cache[video_id]

    # Download all masks
    cache_dir = os.path.join(TEMP_DIR, f"{video_id}_all_masks")
    masks_dir = os.path.join(cache_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)

    zip_path = os.path.join(cache_dir, "all_masks.zip")

    print(f"[3070] Downloading ALL masks (first segment): {all_masks_url}")
    start_time = time.time()

    response = requests.get(all_masks_url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = 0
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)

    download_time = time.time() - start_time
    size_kb = total_size / 1024
    print(f"[3070] Downloaded ALL masks: {size_kb:.2f} KB in {download_time:.2f}s")

    # Extract all masks
    print(f"[3070] Extracting all masks...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.startswith('masks/'):
                data = zf.read(name)
                filename = os.path.basename(name)
                if filename:
                    with open(os.path.join(masks_dir, filename), 'wb') as f:
                        f.write(data)

    # Clean up zip
    os.remove(zip_path)

    num_masks = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    print(f"[3070] Cached {num_masks} masks for video {video_id}")

    # Cache for subsequent segments
    _all_masks_cache[video_id] = masks_dir
    return masks_dir


# Global cache for video frames (download once per video, reuse for all segments)
_video_frames_cache = {}  # video_id -> frames_dir path


def download_video_and_extract_all_frames_cached(video_id, video_url):
    """
    Download video ONCE and extract ALL frames. Cache for all segments.
    Returns frames_dir path (cached on subsequent calls).
    """
    global _video_frames_cache

    # Check cache first
    if video_id in _video_frames_cache:
        cached_dir = _video_frames_cache[video_id]
        if os.path.exists(cached_dir):
            num_frames = len([f for f in os.listdir(cached_dir) if f.endswith('.png')])
            print(f"[3070] Using CACHED frames for {video_id} ({num_frames} frames)")
            return cached_dir
        else:
            del _video_frames_cache[video_id]

    # Download video and extract ALL frames
    cache_dir = os.path.join(TEMP_DIR, f"{video_id}_all_frames")
    frames_dir = os.path.join(cache_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    video_path = os.path.join(cache_dir, "video.mp4")

    print(f"[3070] Downloading video (ONCE for all segments): {video_url}")
    start_time = time.time()

    response = requests.get(video_url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = 0
    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)

    download_time = time.time() - start_time
    size_mb = total_size / (1024 * 1024)
    speed = size_mb / download_time if download_time > 0 else 0
    print(f"[3070] Downloaded video: {size_mb:.2f} MB in {download_time:.2f}s ({speed:.2f} MB/s)")

    # Extract ALL frames using ffmpeg NVDEC (GPU-accelerated, 5-8x faster!)
    print(f"[3070] Extracting ALL frames using NVDEC...")
    extract_start = time.time()

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-hwaccel_output_format', 'cuda',
        '-i', video_path,
        '-vf', 'hwdownload,format=bgr24',
        '-vsync', '0',
        '-start_number', '0',
        os.path.join(frames_dir, '%04d.png')
    ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg NVDEC failed: {result.stderr}")

    # Count extracted frames
    frame_idx = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    extract_time = time.time() - extract_start
    fps = frame_idx / extract_time if extract_time > 0 else 0
    print(f"[3070] NVDEC extracted {frame_idx} frames in {extract_time:.2f}s ({fps:.0f} fps)")

    # Keep video for potential re-use, or remove to save space
    os.remove(video_path)

    print(f"[3070] Cached {frame_idx} frames for video {video_id}")

    # Cache for subsequent segments
    _video_frames_cache[video_id] = frames_dir
    return frames_dir


def download_masks_only(masks_url, work_dir):
    """
    Download masks-only zip from Cloudflare (legacy per-segment).
    Returns masks_dir path.
    """
    zip_path = os.path.join(work_dir, "masks.zip")
    masks_dir = os.path.join(work_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)

    print(f"[3070] Downloading masks: {masks_url}")
    start_time = time.time()

    response = requests.get(masks_url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = 0
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total_size += len(chunk)

    download_time = time.time() - start_time
    size_kb = total_size / 1024
    print(f"[3070] Downloaded masks: {size_kb:.2f} KB in {download_time:.2f}s")

    # Extract masks
    print(f"[3070] Extracting masks...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.startswith('masks/'):
                data = zf.read(name)
                filename = os.path.basename(name)
                if filename:
                    with open(os.path.join(masks_dir, filename), 'wb') as f:
                        f.write(data)

    # Clean up zip
    os.remove(zip_path)

    num_masks = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
    print(f"[3070] Extracted {num_masks} masks")

    return masks_dir


# YOLO detector for crop calculation (lazy load)
_yolo_model = None

def get_yolo_model():
    """Get YOLO model (lazy initialization)"""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = os.path.join(SCRIPT_DIR, 'runs', 'detect', 'new_sora_watermark', 'weights', 'best.pt')
        if not os.path.exists(model_path):
            # Try Docker path
            model_path = '/app/runs/detect/new_sora_watermark/weights/best.pt'
        print(f"[3070] Loading YOLO model from: {model_path}")
        _yolo_model = YOLO(model_path)
        print(f"[3070] YOLO model loaded")
    return _yolo_model


def detect_watermark_bbox_yolo(frames_array, sample_count=5):
    """
    Use YOLO to detect watermark bbox from sample frames.
    Returns (x1, y1, x2, y2) or None if no watermark found.
    """
    if len(frames_array) == 0:
        return None

    model = get_yolo_model()

    # Sample frames evenly
    indices = np.linspace(0, len(frames_array) - 1, min(sample_count, len(frames_array)), dtype=int)
    sample_frames = [frames_array[i] for i in indices]

    all_boxes = []
    for frame in sample_frames:
        results = model(frame, verbose=False)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    all_boxes.append([x1, y1, x2, y2])

    if not all_boxes:
        return None

    # Get union of all detected boxes
    all_boxes = np.array(all_boxes)
    x1 = int(np.min(all_boxes[:, 0]))
    y1 = int(np.min(all_boxes[:, 1]))
    x2 = int(np.max(all_boxes[:, 2]))
    y2 = int(np.max(all_boxes[:, 3]))

    print(f"[3070] YOLO detected watermark at ({x1},{y1})-({x2},{y2})")
    return (x1, y1, x2, y2)


def get_bbox_from_masks(masks_array, frame_height=None, frame_width=None, padding_ratio=0.2):
    """
    Fast mask-based bbox detection (5ms vs 200ms YOLO).
    4090 already ran YOLO detection - just use the mask boundaries!

    Args:
        masks_array: List of mask numpy arrays
        frame_height: Frame height for clamping (optional)
        frame_width: Frame width for clamping (optional)
        padding_ratio: Padding to add around bbox (default 20%)

    Returns:
        (x, y, w, h) tuple or None if no mask content found
    """
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    for mask in masks_array:
        if mask.max() > 0:
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

    # Check if we found any mask content
    if min_x == float('inf') or max_x == 0:
        return None

    # Add padding
    width = max_x - min_x
    height = max_y - min_y
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)

    # Calculate padded bbox
    x1 = max(0, min_x - pad_x)
    y1 = max(0, min_y - pad_y)
    x2 = max_x + pad_x
    y2 = max_y + pad_y

    # Clamp to frame dimensions if provided
    if frame_width:
        x2 = min(x2, frame_width)
    if frame_height:
        y2 = min(y2, frame_height)

    return (x1, y1, x2 - x1, y2 - y1)


# ============================================================================
# VIDEO ASSEMBLY
# ============================================================================

def assemble_final_video(video_id, total_segments, redis_client):
    """
    Combine all segment frames into final video.
    Runs automatically when last segment completes.
    """
    print(f"\n[ASSEMBLY] Starting video assembly for {video_id}...")

    # Get fps from video metadata
    fps = 30  # Default
    metadata_json = redis_client.get(f'video_metadata:{video_id}')
    if metadata_json:
        metadata = json.loads(metadata_json)
        fps = metadata.get('fps', 30)
        print(f"[ASSEMBLY] FPS from metadata: {fps}")

    # Collect all frames from all segments in order
    all_frames = []

    for seg_idx in range(total_segments):
        seg_path = redis_client.hget(f'segment_results:{video_id}', seg_idx)
        if seg_path:
            seg_path = seg_path.decode() if isinstance(seg_path, bytes) else seg_path
            # Get frames from this segment
            if os.path.exists(seg_path):
                frames = sorted([
                    os.path.join(seg_path, f)
                    for f in os.listdir(seg_path)
                    if f.endswith('.png')
                ])
                all_frames.extend(frames)
                print(f"[ASSEMBLY] Segment {seg_idx}: {len(frames)} frames")
            else:
                print(f"[ASSEMBLY] Segment {seg_idx}: path not found: {seg_path}")

    if not all_frames:
        print(f"[ASSEMBLY ERROR] No frames found!")
        return None

    print(f"[ASSEMBLY] Total frames: {len(all_frames)}")

    # Create output video using ffmpeg for web-compatible h264 encoding
    output_path = os.path.join(RESULT_DIR, f"{video_id}_final.mp4")

    # Create a temporary file listing all frames in order
    frame_list_path = os.path.join(RESULT_DIR, f"{video_id}_frames.txt")
    with open(frame_list_path, 'w') as f:
        for frame_path in all_frames:
            # ffmpeg concat demuxer format: file 'path'
            f.write(f"file '{frame_path}'\n")
            f.write(f"duration {1/fps}\n")

    print(f"[ASSEMBLY] Created frame list: {frame_list_path}")

    # Use ffmpeg with NVENC for GPU-accelerated h264 encoding (10-15x faster!)
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', frame_list_path,
        '-c:v', 'h264_nvenc',     # GPU encoder (was libx264)
        '-preset', 'p7',          # Highest quality NVENC preset
        '-profile:v', 'main',     # Browser-compatible H.264 profile
        '-level', '4.1',          # Compatible with most devices
        '-rc', 'vbr',             # Variable bitrate mode
        '-cq', '18',              # Constant quality (CRF-like)
        '-pix_fmt', 'yuv420p',    # Required for browser compatibility
        '-movflags', '+faststart',  # Enables streaming
        output_path
    ]

    print(f"[ASSEMBLY] Running ffmpeg: {' '.join(ffmpeg_cmd)}")

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ASSEMBLY ERROR] ffmpeg failed: {result.stderr}")
        # Fallback to cv2 if ffmpeg fails
        print("[ASSEMBLY] Falling back to cv2.VideoWriter...")
        first_frame = cv2.imread(all_frames[0])
        if first_frame is None:
            print(f"[ASSEMBLY ERROR] Could not read first frame!")
            return None
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for i, frame_path in enumerate(all_frames):
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
            if (i + 1) % 100 == 0:
                print(f"[ASSEMBLY] Written {i + 1}/{len(all_frames)} frames...")
        out.release()

    # Clean up frame list
    if os.path.exists(frame_list_path):
        os.remove(frame_list_path)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[ASSEMBLY] Video saved: {output_path} ({size_mb:.2f} MB)")

        # Store final path in Redis
        redis_client.set(f'final_video:{video_id}', output_path)

        # TODO: Upload to server (commented for now - enable when ready)
        # upload_final_video_to_b2(video_id, output_path)

        return output_path
    else:
        print(f"[ASSEMBLY ERROR] Failed to create video!")
        return None


# ============================================================================
# CELERY TASKS
# ============================================================================

@celery.task(bind=True, name='worker.process_segment')
def process_segment_task(self, video_id, segment_index, start_frame, end_frame,
                         video_url=None, all_masks_url=None, total_segments=None,
                         masks_url=None, cloudflare_url=None):
    """
    Process a single video segment - runs on 3070

    Args:
        video_id: Video identifier
        segment_index: Index of this segment
        start_frame: First frame of segment
        end_frame: Last frame of segment
        video_url: URL to download original video (3070 extracts frames)
        all_masks_url: URL to download ALL masks (download once, cache for all segments)
        total_segments: Total number of segments for this video
        masks_url: Legacy - URL to download masks-only zip from Cloudflare (per-segment)
        cloudflare_url: Legacy - URL to download segment zip (frames+masks) from Cloudflare

    Returns:
        Dict with status and output path
    """
    print(f"\n{'='*60}")
    print(f"[3070 WORKER] Processing segment {segment_index}")
    print(f"[3070 WORKER] Video: {video_id}, Frames: {start_frame}-{end_frame}")
    print(f"{'='*60}\n")

    # Update status in Redis
    redis_client = celery.backend.client
    status_key = f'segment_status:{video_id}:{segment_index}'
    redis_client.set(status_key, 'processing')

    try:
        # Update Celery state
        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Loading metadata'
        })

        # ====================================================================
        # STEP 1: Get video metadata from Redis
        # ====================================================================

        metadata_json = redis_client.get(f'video_metadata:{video_id}')
        if not metadata_json:
            raise Exception(f"Video metadata not found: {video_id}")

        metadata = json.loads(metadata_json)

        frames_dir = metadata['frames_dir']
        masks_dir = metadata['masks_dir']
        fps = metadata['fps']
        width = metadata['width']
        height = metadata['height']
        total_segments = metadata.get('total_segments', len(metadata['segments']))
        api_base = metadata.get('api_base')

        print(f"[3070] Metadata loaded: {width}x{height}, fps={fps}")

        # ====================================================================
        # STEP 2: Prepare segment directories
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Preparing directories'
        })

        # Create local working directories
        work_dir = os.path.join(TEMP_DIR, f"{video_id}_seg{segment_index}")
        seg_frames_dir = os.path.join(work_dir, 'frames')
        seg_masks_dir = os.path.join(work_dir, 'masks')
        seg_output_dir = os.path.join(work_dir, 'output')

        # Clean up any existing work directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        os.makedirs(seg_frames_dir, exist_ok=True)
        os.makedirs(seg_masks_dir, exist_ok=True)
        os.makedirs(seg_output_dir, exist_ok=True)

        # ====================================================================
        # STEP 3: Download/copy segment frames and masks
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Loading frames'
        })

        num_frames = end_frame - start_frame + 1
        print(f"[3070] Loading {num_frames} frames for segment...")

        # PRIORITY 1: NEW BATCH ARCHITECTURE - Video + masks BOTH cached
        # Downloads ONCE on first segment, all others use cache
        if video_url and all_masks_url:
            print(f"[3070] BATCH ARCHITECTURE: Video + ALL masks (BOTH CACHED)")
            print(f"[3070]   video_url: {video_url}")
            print(f"[3070]   all_masks_url: {all_masks_url}")
            print(f"[3070]   Segment {segment_index + 1}/{total_segments or '?'}")

            # Get cached frames (downloads video ONCE, extracts ALL frames, cached for all segments)
            all_frames_dir = download_video_and_extract_all_frames_cached(video_id, video_url)

            # Get cached masks (downloads once on first segment, cached for rest)
            all_masks_dir = download_all_masks_cached(video_id, all_masks_url)

            # Copy just this segment's frames from the cache
            seg_frames_dir = os.path.join(work_dir, 'frames')
            os.makedirs(seg_frames_dir, exist_ok=True)

            for frame_idx in range(start_frame, end_frame + 1):
                # Source: original frame index in all_frames
                src = os.path.join(all_frames_dir, f"{frame_idx:04d}.png")
                # Dest: local index within segment
                local_idx = frame_idx - start_frame
                dst = os.path.join(seg_frames_dir, f"{local_idx:04d}.png")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    print(f"[3070] WARNING: Frame {frame_idx} not found in cache!")

            # Copy just this segment's masks from the cache
            seg_masks_dir = os.path.join(work_dir, 'masks')
            os.makedirs(seg_masks_dir, exist_ok=True)

            for frame_idx in range(start_frame, end_frame + 1):
                # Source: original frame index in all_masks
                src = os.path.join(all_masks_dir, f"{frame_idx:04d}.png")
                # Dest: local index within segment
                local_idx = frame_idx - start_frame
                dst = os.path.join(seg_masks_dir, f"{local_idx:04d}.png")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    # No mask = no watermark on this frame
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.imwrite(dst, mask)

        # PRIORITY 2: Legacy per-segment masks (old architecture)
        elif video_url and masks_url:
            print(f"[3070] LEGACY: Download video + per-segment masks")
            print(f"[3070]   video_url: {video_url}")
            print(f"[3070]   masks_url: {masks_url}")

            # Download video and extract only the frames we need
            seg_frames_dir = download_video_and_extract_frames(
                video_url, work_dir, start_frame, end_frame
            )

            # Download per-segment masks
            seg_masks_dir = download_masks_only(masks_url, work_dir)

        # PRIORITY 2: Legacy - Download from Cloudflare (frames+masks zip)
        elif cloudflare_url:
            print(f"[3070] LEGACY: Using Cloudflare URL (frames+masks): {cloudflare_url}")
            seg_frames_dir, seg_masks_dir = download_and_extract_from_cloudflare(cloudflare_url, work_dir)

        # PRIORITY 3: Direct access to local files
        elif os.path.exists(frames_dir):
            print(f"[3070] Using local frames: {frames_dir}")
            for frame_idx in range(start_frame, end_frame + 1):
                src = os.path.join(frames_dir, f"{frame_idx:04d}.png")
                dst = os.path.join(seg_frames_dir, f"{frame_idx - start_frame:04d}.png")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    print(f"[3070] WARNING: Frame not found: {src}")

            # Copy masks
            for frame_idx in range(start_frame, end_frame + 1):
                src = os.path.join(masks_dir, f"{frame_idx:04d}.png")
                dst = os.path.join(seg_masks_dir, f"{frame_idx - start_frame:04d}.png")
                if os.path.exists(src):
                    shutil.copy(src, dst)
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.imwrite(dst, mask)

        # PRIORITY 4: Download from API
        elif api_base:
            print(f"[3070] Using API: {api_base}")
            for frame_idx in range(start_frame, end_frame + 1):
                url = f"{api_base}/frames/{video_id}/{frame_idx:04d}.png"
                dst = os.path.join(seg_frames_dir, f"{frame_idx - start_frame:04d}.png")
                download_file(url, dst)

            for frame_idx in range(start_frame, end_frame + 1):
                url = f"{api_base}/masks/{video_id}/{frame_idx:04d}.png"
                dst = os.path.join(seg_masks_dir, f"{frame_idx - start_frame:04d}.png")
                download_file(url, dst)

        else:
            raise Exception("No video_url+masks_url, no cloudflare_url, no local access, and no API base URL")

        print(f"[3070] Loaded {num_frames} frames and masks")

        # ====================================================================
        # STEP 4: Run ProPainter (IN-MEMORY for 15x speedup)
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Loading frames into memory'
        })

        # Load frames into memory for fast processing
        print(f"[3070] Loading frames into memory...")
        frames_array = []
        for f in sorted(os.listdir(seg_frames_dir)):
            if f.endswith('.png'):
                img = cv2.imread(os.path.join(seg_frames_dir, f))
                frames_array.append(np.ascontiguousarray(img))

        # Load masks into memory
        print(f"[3070] Loading masks into memory...")
        masks_array = []
        for f in sorted(os.listdir(seg_masks_dir)):
            if f.endswith('.png'):
                mask = cv2.imread(os.path.join(seg_masks_dir, f), cv2.IMREAD_GRAYSCALE)
                masks_array.append(np.ascontiguousarray(mask))

        print(f"[3070] Loaded {len(frames_array)} frames, {len(masks_array)} masks into memory")

        # ====================================================================
        # STEP 4b: CROP TO WATERMARK REGION (MASK-BASED - 16x SPEEDUP!)
        # Use SAM2 mask bbox directly (more precise than YOLO, no union issue!)
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Computing watermark region from masks'
        })

        # Use mask-based detection (faster and more accurate than YOLO!)
        # YOLO samples 5 frames and gets UNION = large bbox
        # Masks give EXACT per-frame watermark position = small bbox
        print(f"[3070] Computing watermark region from SAM2 masks (faster than YOLO)...")
        min_x, min_y = width, height
        max_x, max_y = 0, 0

        for mask in masks_array:
            if mask.max() > 0:  # Faster than np.sum(mask > 127)
                coords = cv2.findNonZero(mask.astype(np.uint8))
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)

        use_cropping = (max_x > min_x and max_y > min_y)

        if use_cropping:
            print(f"[3070] Mask bbox: ({min_x},{min_y})-({max_x},{max_y})")

        if use_cropping:
            bbox = [min_x, min_y, max_x, max_y]
            # Use 0.15 padding like RUN_SAM2_LOCAL (tighter crop = faster)
            crop_x, crop_y, crop_w, crop_h = calculate_crop_region(
                bbox, width, height, padding_ratio=0.15, min_size=128
            )

            # Calculate speedup estimate
            orig_pixels = width * height
            crop_pixels = crop_w * crop_h
            speedup_est = orig_pixels / crop_pixels if crop_pixels > 0 else 1

            print(f"[3070] Crop bbox: ({min_x},{min_y})-({max_x},{max_y})")
            print(f"[3070] Crop region: ({crop_x},{crop_y}) {crop_w}x{crop_h}")
            print(f"[3070] Estimated speedup: {speedup_est:.1f}x ({orig_pixels} -> {crop_pixels} pixels)")

            # Crop all frames and masks
            cropped_frames = [f[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy() for f in frames_array]
            cropped_masks = [m[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy() for m in masks_array]

            # Make contiguous for GPU
            cropped_frames = [np.ascontiguousarray(f) for f in cropped_frames]
            cropped_masks = [np.ascontiguousarray(m) for m in cropped_masks]

            print(f"[3070] Cropped to {len(cropped_frames)} frames at {crop_w}x{crop_h}")

            # Use cropped data for processing
            process_frames = cropped_frames
            process_masks = cropped_masks
        else:
            print(f"[3070] No watermark region found, processing full frame")
            process_frames = frames_array
            process_masks = masks_array
            crop_x, crop_y, crop_w, crop_h = 0, 0, width, height

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Running ProPainter'
        })

        # Calculate optimal parameters
        neighbor_length = 10
        ref_stride = 10
        dynamic_subvideo = get_dynamic_subvideo_length(crop_w, crop_h)  # Use cropped dimensions!

        # Use FP16 on 3070 for speed
        use_fp16 = True

        print(f"[3070] Running ProPainter (neighbor={neighbor_length}, ref_stride={ref_stride}, subvideo={dynamic_subvideo})")

        process_start = time.time()

        faster_propainter_pipeline(
            video=seg_frames_dir,
            mask=seg_masks_dir,
            output=seg_output_dir,
            resize_ratio=1.0,
            mask_dilation=4,
            ref_stride=ref_stride,
            neighbor_length=neighbor_length,
            subvideo_length=dynamic_subvideo,
            raft_iter=10,
            mode="video_inpainting",
            save_frames=True,
            fp16=use_fp16,
            frames_array=process_frames,  # CROPPED!
            masks_array=process_masks     # CROPPED!
        )

        process_time = time.time() - process_start
        ms_per_frame = (process_time / num_frames) * 1000
        print(f"[3070] ProPainter complete: {process_time:.2f}s ({ms_per_frame:.2f}ms/frame)")

        # Clear GPU cache to prevent VRAM accumulation
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Find output frames
        propainter_output = os.path.join(seg_output_dir, os.path.basename(seg_frames_dir), "frames")
        if not os.path.exists(propainter_output):
            # Try alternative path
            propainter_output = os.path.join(seg_output_dir, "frames")

        if not os.path.exists(propainter_output):
            raise Exception(f"ProPainter output not found at {propainter_output}")

        output_frames = sorted([f for f in os.listdir(propainter_output) if f.endswith('.png')])
        print(f"[3070] Generated {len(output_frames)} output frames")

        # ====================================================================
        # STEP 4c: PASTE CROPPED OUTPUT BACK INTO FULL FRAMES
        # ====================================================================

        if use_cropping:
            print(f"[3070] Pasting cropped output back into full frames...")
            for i, fname in enumerate(output_frames):
                # Read cropped output frame
                output_path = os.path.join(propainter_output, fname)
                cropped_output = cv2.imread(output_path)

                if cropped_output is None:
                    continue

                # Get original full frame
                if i < len(frames_array):
                    full_frame = frames_array[i].copy()

                    # Paste cropped output back into the crop region
                    full_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cropped_output

                    # Overwrite the output file with full-resolution frame
                    cv2.imwrite(output_path, full_frame)

            print(f"[3070] Merged {len(output_frames)} frames back to full resolution ({width}x{height})")

        # ====================================================================
        # STEP 5: Save results LOCALLY (no uploads until all done!)
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Saving results locally'
        })

        # Save results to local directory ONLY (no B2 upload!)
        result_dir = os.path.join(RESULT_DIR, f"{video_id}_segments", f"segment_{segment_index}")
        os.makedirs(result_dir, exist_ok=True)

        for frame_file in output_frames:
            src = os.path.join(propainter_output, frame_file)
            dst = os.path.join(result_dir, frame_file)
            shutil.copy(src, dst)

        print(f"[3070] Results saved locally: {result_dir}")

        # NO B2 UPLOAD - results stay local until all segments done!
        # Windows script will assemble from local results

        # Notify Redis that segment is complete
        redis_client.set(status_key, 'completed')

        # Store LOCAL path in Redis (not URL!) for assembly
        redis_client.hset(f'segment_results:{video_id}', segment_index, result_dir)

        # Store total_segments for assembly script
        if total_segments:
            redis_client.set(f'total_segments:{video_id}', total_segments)

        # Publish completion event (for real-time updates)
        completion_data = {
            'video_id': video_id,
            'segment_index': segment_index,
            'output_path': result_dir,
            'num_frames': len(output_frames),
            'process_time': process_time,
            'total_segments': total_segments
        }
        redis_client.publish(f'segment_complete:{video_id}', json.dumps(completion_data))

        # ====================================================================
        # CHECK IF ALL SEGMENTS COMPLETE - AUTO-ASSEMBLE + UPLOAD
        # ====================================================================

        if total_segments:
            completed_count = redis_client.hlen(f'segment_results:{video_id}')
            print(f"[3070] Segment completion: {completed_count}/{total_segments}")

            if completed_count == total_segments:
                print(f"\n[3070] ALL SEGMENTS COMPLETE! Starting assembly + upload...")

                try:
                    # Assemble final video
                    final_path = assemble_final_video(video_id, total_segments, redis_client)

                    if final_path and os.path.exists(final_path):
                        # Upload to B2
                        final_url = upload_final_video_to_b2(video_id, final_path)

                        if final_url:
                            # Store final URL in Redis
                            redis_client.set(f'final_url:{video_id}', final_url)

                            # Publish video complete event
                            redis_client.publish(f'video_complete:{video_id}', json.dumps({
                                'video_id': video_id,
                                'url': final_url,
                                'total_segments': total_segments
                            }))

                            print(f"\n[3070] FINAL VIDEO UPLOADED!")
                            print(f"   URL: {final_url}")
                        else:
                            print(f"[3070] WARNING: Upload failed, video saved locally at {final_path}")
                    else:
                        print(f"[3070] WARNING: Assembly failed")

                except Exception as assembly_error:
                    print(f"[3070] ERROR during assembly/upload: {assembly_error}")

        # ====================================================================
        # DONE
        # ====================================================================

        # Cleanup work directory
        shutil.rmtree(work_dir)

        result = {
            'status': 'success',
            'video_id': video_id,
            'segment_index': segment_index,
            'output_path': result_dir,
            'num_frames': len(output_frames),
            'process_time': process_time,
            'ms_per_frame': ms_per_frame,
            'total_segments': total_segments
        }

        print(f"\n[3070] Segment {segment_index} complete!")
        print(f"   Output: {result_dir} (local)")
        print(f"   Frames: {len(output_frames)}")
        print(f"   Time: {process_time:.2f}s")

        return result

    except Exception as e:
        # Mark as failed
        redis_client.set(status_key, 'failed')

        error_msg = str(e)
        print(f"\n[3070 ERROR] Segment {segment_index} failed: {error_msg}")

        return {
            'status': 'failed',
            'video_id': video_id,
            'segment_index': segment_index,
            'error': error_msg
        }


@celery.task(bind=True, name='worker.process_batch')
def process_batch_task(self, video_id, video_url, all_masks_url, total_frames, fps=30, width=None, height=None):
    """
    BATCH MODE: Process ENTIRE video in ONE ProPainter call.
    Matches RUN_SAM2_LOCAL.py performance (~15s vs ~32s for segments).

    Key optimizations:
    1. NO YOLO - use mask-based bbox (5ms vs 200ms)
    2. ONE ProPainter call for ALL frames (not 4 separate calls)
    3. RAFT optical flow computed once across ALL frames
    4. subvideo_length=120 for optimal batching

    Args:
        video_id: Video identifier
        video_url: URL to download original video
        all_masks_url: URL to download ALL masks zip
        total_frames: Total number of frames
        fps: Video FPS (default 30)
        width: Frame width (optional, auto-detected)
        height: Frame height (optional, auto-detected)

    Returns:
        Dict with status and output path
    """
    print(f"\n{'='*60}")
    print(f"[3070 BATCH] Processing ENTIRE video in ONE call")
    print(f"[3070 BATCH] Video: {video_id}, Frames: {total_frames}")
    print(f"{'='*60}\n")

    # Update status in Redis
    redis_client = celery.backend.client
    status_key = f'batch_status:{video_id}'
    redis_client.set(status_key, 'processing')

    try:
        batch_start = time.time()

        # Update Celery state
        self.update_state(state='PROCESSING', meta={
            'status': 'Downloading video and masks'
        })

        # ====================================================================
        # STEP 1: Download video and extract ALL frames
        # ====================================================================

        work_dir = os.path.join(TEMP_DIR, f"{video_id}_batch")
        frames_dir = os.path.join(work_dir, 'frames')
        output_dir = os.path.join(work_dir, 'output')

        # Clean up any existing work directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Download video
        video_path = os.path.join(work_dir, "video.mp4")
        print(f"[3070 BATCH] Downloading video: {video_url}")
        download_start = time.time()

        response = requests.get(video_url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = 0
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

        download_time = time.time() - download_start
        size_mb = total_size / (1024 * 1024)
        print(f"[3070 BATCH] Downloaded video: {size_mb:.2f} MB in {download_time:.2f}s")

        # Extract ALL frames
        print(f"[3070 BATCH] Extracting ALL {total_frames} frames...")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")

        # Get video properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

        if width is None:
            width = actual_width
        if height is None:
            height = actual_height

        print(f"[3070 BATCH] Video: {width}x{height} @ {actual_fps}fps")

        frames_array = []
        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_array.append(np.ascontiguousarray(frame))
            frame_idx += 1

        cap.release()
        os.remove(video_path)  # Clean up video file

        print(f"[3070 BATCH] Extracted {len(frames_array)} frames into memory")

        # ====================================================================
        # STEP 2: Download ALL masks (cached)
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'status': 'Loading masks'
        })

        all_masks_dir = download_all_masks_cached(video_id, all_masks_url)

        # Load ALL masks into memory
        print(f"[3070 BATCH] Loading ALL masks into memory...")
        masks_array = []
        for i in range(total_frames):
            mask_path = os.path.join(all_masks_dir, f"{i:04d}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                masks_array.append(np.ascontiguousarray(mask))
            else:
                # No mask = no watermark on this frame
                masks_array.append(np.zeros((height, width), dtype=np.uint8))

        print(f"[3070 BATCH] Loaded {len(masks_array)} masks into memory")

        # ====================================================================
        # STEP 3: Get bbox from masks (NOT YOLO - 5ms vs 200ms!)
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'status': 'Calculating crop region'
        })

        print(f"[3070 BATCH] Using FAST mask-based bbox (skipping YOLO)...")
        bbox_start = time.time()
        bbox = get_bbox_from_masks(masks_array, height, width, padding_ratio=0.2)
        bbox_time = (time.time() - bbox_start) * 1000
        print(f"[3070 BATCH] Mask bbox calculated in {bbox_time:.1f}ms (YOLO would be ~200ms)")

        if bbox:
            crop_x, crop_y, crop_w, crop_h = bbox

            # Calculate speedup estimate
            orig_pixels = width * height
            crop_pixels = crop_w * crop_h
            speedup_est = orig_pixels / crop_pixels if crop_pixels > 0 else 1

            print(f"[3070 BATCH] Crop region: ({crop_x},{crop_y}) {crop_w}x{crop_h}")
            print(f"[3070 BATCH] Estimated speedup: {speedup_est:.1f}x ({orig_pixels} -> {crop_pixels} pixels)")

            # Crop ALL frames and masks
            cropped_frames = [f[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy() for f in frames_array]
            cropped_masks = [m[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy() for m in masks_array]

            # Make contiguous for GPU
            cropped_frames = [np.ascontiguousarray(f) for f in cropped_frames]
            cropped_masks = [np.ascontiguousarray(m) for m in cropped_masks]

            print(f"[3070 BATCH] Cropped ALL {len(cropped_frames)} frames to {crop_w}x{crop_h}")

            process_frames = cropped_frames
            process_masks = cropped_masks
            use_cropping = True
        else:
            print(f"[3070 BATCH] No watermark region found, processing full frames")
            process_frames = frames_array
            process_masks = masks_array
            crop_x, crop_y, crop_w, crop_h = 0, 0, width, height
            use_cropping = False

        # ====================================================================
        # STEP 4: ONE ProPainter call for ALL frames (THE BIG SPEEDUP!)
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'status': f'Running ProPainter on ALL {len(process_frames)} frames'
        })

        # Use optimal subvideo_length=120 for cropped regions (like RUN_SAM2_LOCAL)
        dynamic_subvideo = 120  # Optimal for cropped regions
        neighbor_length = 10
        ref_stride = 10
        use_fp16 = True

        print(f"[3070 BATCH] Running ProPainter (subvideo={dynamic_subvideo}, ALL {len(process_frames)} frames)")
        print(f"[3070 BATCH] This is ONE call instead of 4 segment calls!")

        process_start = time.time()

        faster_propainter_pipeline(
            video=frames_dir,  # Dummy path, we use frames_array
            mask=work_dir,     # Dummy path, we use masks_array
            output=output_dir,
            resize_ratio=1.0,
            mask_dilation=4,
            ref_stride=ref_stride,
            neighbor_length=neighbor_length,
            subvideo_length=dynamic_subvideo,
            raft_iter=10,
            mode="video_inpainting",
            save_frames=True,
            fp16=use_fp16,
            frames_array=process_frames,  # ALL frames at once!
            masks_array=process_masks     # ALL masks at once!
        )

        process_time = time.time() - process_start
        ms_per_frame = (process_time / len(process_frames)) * 1000
        print(f"[3070 BATCH] ProPainter complete: {process_time:.2f}s ({ms_per_frame:.2f}ms/frame)")

        # Find output frames
        propainter_output = os.path.join(output_dir, os.path.basename(frames_dir), "frames")
        if not os.path.exists(propainter_output):
            propainter_output = os.path.join(output_dir, "frames")

        if not os.path.exists(propainter_output):
            raise Exception(f"ProPainter output not found at {propainter_output}")

        output_frames = sorted([f for f in os.listdir(propainter_output) if f.endswith('.png')])
        print(f"[3070 BATCH] Generated {len(output_frames)} output frames")

        # ====================================================================
        # STEP 5: Paste cropped output back into full frames
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'status': 'Merging results to full resolution'
        })

        if use_cropping:
            print(f"[3070 BATCH] Pasting cropped output back into full frames...")
            for i, fname in enumerate(output_frames):
                output_path = os.path.join(propainter_output, fname)
                cropped_output = cv2.imread(output_path)

                if cropped_output is None:
                    continue

                if i < len(frames_array):
                    full_frame = frames_array[i].copy()
                    full_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cropped_output
                    cv2.imwrite(output_path, full_frame)

            print(f"[3070 BATCH] Merged {len(output_frames)} frames to full resolution ({width}x{height})")

        # ====================================================================
        # STEP 6: Save results locally
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'status': 'Saving results'
        })

        result_dir = os.path.join(RESULT_DIR, f"{video_id}_batch", "frames")
        os.makedirs(result_dir, exist_ok=True)

        for frame_file in output_frames:
            src = os.path.join(propainter_output, frame_file)
            dst = os.path.join(result_dir, frame_file)
            shutil.copy(src, dst)

        print(f"[3070 BATCH] Results saved locally: {result_dir}")

        # Update Redis
        redis_client.set(status_key, 'completed')
        redis_client.set(f'batch_result:{video_id}', result_dir)
        redis_client.set(f'batch_frames:{video_id}', len(output_frames))

        # Also set segment-compatible keys for download_and_assemble.py
        redis_client.set(f'total_segments:{video_id}', 1)
        redis_client.hset(f'segment_results:{video_id}', 0, result_dir)
        redis_client.set(f'segment_status:{video_id}:0', 'completed')

        # Cleanup work directory
        shutil.rmtree(work_dir)

        # Clear masks cache for this video
        if video_id in _all_masks_cache:
            cached_dir = os.path.dirname(_all_masks_cache[video_id])
            if os.path.exists(cached_dir):
                shutil.rmtree(cached_dir)
            del _all_masks_cache[video_id]

        batch_total = time.time() - batch_start

        result = {
            'status': 'success',
            'video_id': video_id,
            'output_path': result_dir,
            'num_frames': len(output_frames),
            'process_time': process_time,
            'total_time': batch_total,
            'ms_per_frame': ms_per_frame,
            'mode': 'batch'
        }

        print(f"\n{'='*60}")
        print(f"[3070 BATCH] COMPLETE!")
        print(f"   Output: {result_dir}")
        print(f"   Frames: {len(output_frames)}")
        print(f"   ProPainter: {process_time:.2f}s")
        print(f"   Total: {batch_total:.2f}s")
        print(f"   Speed: {ms_per_frame:.2f}ms/frame")
        print(f"{'='*60}\n")

        return result

    except Exception as e:
        redis_client.set(status_key, 'failed')
        error_msg = str(e)
        print(f"\n[3070 BATCH ERROR] Failed: {error_msg}")

        return {
            'status': 'failed',
            'video_id': video_id,
            'error': error_msg,
            'mode': 'batch'
        }


@celery.task(bind=True, name='worker.health_check')
def health_check(self):
    """Health check task to verify worker is running"""
    import torch

    return {
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'timestamp': time.time()
    }


# ============================================================================
# WORKER STARTUP
# ============================================================================

@worker_process_init.connect
def init_worker(**kwargs):
    """Initialize worker when it starts"""
    import torch

    print("[3070 WORKER] Initializing processing worker...")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[3070 WORKER] GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("[3070 WORKER] WARNING: No GPU available!")

    print("[3070 WORKER] Ready to process segments!")


if __name__ == '__main__':
    print("Run with: celery -A celery_3070_worker worker -Q processing -c 1")
