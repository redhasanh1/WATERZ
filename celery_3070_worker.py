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
import zipfile
import requests
import numpy as np
from pathlib import Path

# Add python_packages to path
sys.path.insert(0, 'python_packages')

# ============================================================================
# GPU OPTIMIZATIONS FOR RTX 3070
# ============================================================================

# Disable Flash Attention (not supported on RTX 3070)
os.environ['ENABLE_FLASH_ATTENTION'] = '0'
os.environ['TORCH_CUDAGRAPHS'] = '0'
os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'

# Disable torch.compile on Windows
if sys.platform == 'win32':
    os.environ['USE_TORCH_COMPILE_RAFT'] = '0'

import cv2
from celery import Celery

# Import ProPainter
from watermark import pipeline as faster_propainter_pipeline

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

# ============================================================================
# CELERY TASKS
# ============================================================================

@celery.task(bind=True, name='worker.process_segment')
def process_segment_task(self, video_id, segment_index, start_frame, end_frame, cloudflare_url=None):
    """
    Process a single video segment - runs on 3070

    Args:
        video_id: Video identifier
        segment_index: Index of this segment
        start_frame: First frame of segment
        end_frame: Last frame of segment
        cloudflare_url: Optional URL to download segment zip from Cloudflare

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

        # PRIORITY 1: Download from Cloudflare if URL provided
        if cloudflare_url:
            print(f"[3070] Using Cloudflare URL: {cloudflare_url}")
            seg_frames_dir, seg_masks_dir = download_and_extract_from_cloudflare(cloudflare_url, work_dir)

        # PRIORITY 2: Direct access to local files
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

        # PRIORITY 3: Download from API
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
            raise Exception("No cloudflare_url, no local access, and no API base URL")

        print(f"[3070] Loaded {num_frames} frames and masks")

        # ====================================================================
        # STEP 4: Run ProPainter
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Running ProPainter'
        })

        # Calculate optimal parameters
        neighbor_length = 10
        ref_stride = 10
        dynamic_subvideo = get_dynamic_subvideo_length(width, height)

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
            frames_array=None,
            masks_array=None
        )

        process_time = time.time() - process_start
        ms_per_frame = (process_time / num_frames) * 1000
        print(f"[3070] ProPainter complete: {process_time:.2f}s ({ms_per_frame:.2f}ms/frame)")

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
        # STEP 5: Store results and upload to Railway
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Uploading results'
        })

        # Copy results to local directory first
        result_dir = os.path.join(RESULT_DIR, f"{video_id}_segments", f"segment_{segment_index}")
        os.makedirs(result_dir, exist_ok=True)

        for frame_file in output_frames:
            src = os.path.join(propainter_output, frame_file)
            dst = os.path.join(result_dir, frame_file)
            shutil.copy(src, dst)

        # Upload to Railway
        railway_result_url = upload_result_to_railway(video_id, segment_index, propainter_output)

        # Notify Redis that segment is complete
        redis_client.set(status_key, 'completed')

        # Publish completion event (for real-time updates)
        completion_data = {
            'video_id': video_id,
            'segment_index': segment_index,
            'output_path': result_dir,
            'railway_result_url': railway_result_url,
            'num_frames': len(output_frames),
            'process_time': process_time
        }
        redis_client.publish(f'segment_complete:{video_id}', json.dumps(completion_data))

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
            'railway_result_url': railway_result_url,
            'num_frames': len(output_frames),
            'process_time': process_time,
            'ms_per_frame': ms_per_frame
        }

        print(f"\n[3070] Segment {segment_index} complete!")
        print(f"   Output: {result_dir}")
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

@celery.signals.worker_process_init.connect
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
