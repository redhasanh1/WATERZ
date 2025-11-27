"""
FAKE RTX 4090 Detection Server - For Testing Without GPU

This simulates the 4090 detection server without running YOLO.
Creates fake segments based on video length for testing job distribution.
"""

import os
import sys
import json
import time
import uuid
import shutil
import zipfile
import numpy as np
from pathlib import Path

sys.path.insert(0, 'python_packages')

import cv2
from celery import Celery
from celery.signals import worker_process_init
from b2sdk.v2 import B2Api, InMemoryAccountInfo

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Redis configuration
REDIS_URL = None
redis_url_file = os.path.join(SCRIPT_DIR, 'redis_url.txt')
if os.path.exists(redis_url_file):
    with open(redis_url_file, 'r') as f:
        REDIS_URL = f.read().strip()
    print(f"[OK] Loaded Redis URL from {redis_url_file}")

broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL") or REDIS_URL
backend = os.getenv("CELERY_RESULT_BACKEND") or broker

if not broker:
    raise ValueError("Redis URL is required")

print(f"[FAKE 4090] Celery Broker = {broker}")

# Directories
UPLOAD_DIR = os.path.join(SCRIPT_DIR, 'uploads')
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
MASK_DIR = os.path.join(SCRIPT_DIR, 'masks')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# B2 + Cloudflare configuration
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

def upload_segment_to_b2(video_id, segment_idx, frames_dir, masks_dir, start_frame, end_frame):
    """
    Zip segment frames+masks and upload to B2.
    Returns Cloudflare URL for download.
    """
    # Create zip file
    zip_filename = f"{video_id}_seg{segment_idx}.zip"
    zip_path = os.path.join(TEMP_DIR, zip_filename)

    print(f"[B2] Creating zip for segment {segment_idx}...")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add frames for this segment
        for frame_idx in range(start_frame, end_frame + 1):
            frame_file = f"{frame_idx:04d}.png"
            frame_path = os.path.join(frames_dir, frame_file)
            if os.path.exists(frame_path):
                # Store with local index (0000, 0001, etc.)
                local_idx = frame_idx - start_frame
                zf.write(frame_path, f"frames/{local_idx:04d}.png")

        # Add masks for this segment
        for frame_idx in range(start_frame, end_frame + 1):
            mask_file = f"{frame_idx:04d}.png"
            mask_path = os.path.join(masks_dir, mask_file)
            if os.path.exists(mask_path):
                local_idx = frame_idx - start_frame
                zf.write(mask_path, f"masks/{local_idx:04d}.png")

    zip_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[B2] Zip created: {zip_filename} ({zip_size:.2f} MB)")

    # Upload to B2
    bucket = get_b2_bucket()
    b2_path = f"segments/{video_id}/{zip_filename}"

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
    return cloudflare_url

# ============================================================================
# CELERY SETUP
# ============================================================================

celery = Celery('fake_detection_4090', broker=broker, backend=backend)

celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    worker_prefetch_multiplier=1,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
    # Route detection tasks to detection queue
    task_routes={
        'detection.*': {'queue': 'detection'},
    },
    task_default_queue='detection',
)

# ============================================================================
# FAKE DETECTION TASK
# ============================================================================

@celery.task(bind=True, name='detection.detect_video')
def detect_video_task(self, video_path, mode='yolo', click_points=None, api_base=None):
    """
    FAKE detection task - creates mock segments without running YOLO
    """
    print(f"\n{'='*60}")
    print(f"[FAKE 4090] Processing: {video_path}")
    print(f"[FAKE 4090] Mode: {mode} (SIMULATED)")
    print(f"{'='*60}\n")

    video_id = uuid.uuid4().hex[:8]
    base_name = Path(video_path).stem

    self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Loading video'})

    # ========================================================================
    # STEP 1: Read video info (no GPU needed)
    # ========================================================================

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"[FAKE 4090] Video: {width}x{height} @ {fps} fps ({total_frames} frames)")

    # ========================================================================
    # STEP 2: Extract frames (CPU only)
    # ========================================================================

    print(f"[FAKE 4090] Extracting {total_frames} frames...")
    self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'Extracting frames'})

    frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_originals")
    masks_dir = os.path.join(MASK_DIR, f"{base_name}_{video_id}")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:04d}.png"), frame)
        frame_idx += 1

    cap.release()
    frames_extracted = frame_idx
    print(f"[FAKE 4090] Extracted {frames_extracted} frames")

    # ========================================================================
    # STEP 3: Create FAKE segments (simulate watermark detection)
    # ========================================================================

    print(f"[FAKE 4090] Creating FAKE segments (simulated detection)...")
    self.update_state(state='PROCESSING', meta={'progress': 40, 'status': 'Fake detection'})

    # Simulate: Create 3-5 fake segments spread across the video
    num_segments = min(5, max(1, frames_extracted // 50))
    segment_length = frames_extracted // (num_segments + 1)

    segments = []
    for i in range(num_segments):
        start = (i + 1) * segment_length - 10
        end = start + 30
        start = max(0, start)
        end = min(frames_extracted - 1, end)
        segments.append((start, end))

    print(f"[FAKE 4090] Created {len(segments)} FAKE segments: {segments}")

    # ========================================================================
    # STEP 4: Create FAKE masks (white rectangles)
    # ========================================================================

    print(f"[FAKE 4090] Creating FAKE masks...")
    self.update_state(state='PROCESSING', meta={'progress': 60, 'status': 'Creating masks'})

    # Create masks for segment frames
    frames_with_watermark = 0
    for seg_start, seg_end in segments:
        for frame_idx in range(seg_start, seg_end + 1):
            # Create fake mask with white rectangle in corner
            mask = np.zeros((height, width), dtype=np.uint8)
            # Fake watermark in top-right corner
            mask[10:60, width-110:width-10] = 255
            cv2.imwrite(os.path.join(masks_dir, f"{frame_idx:04d}.png"), mask)
            frames_with_watermark += 1

    # Create empty masks for non-watermark frames
    for frame_idx in range(frames_extracted):
        mask_path = os.path.join(masks_dir, f"{frame_idx:04d}.png")
        if not os.path.exists(mask_path):
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.imwrite(mask_path, mask)

    print(f"[FAKE 4090] {frames_with_watermark} frames with fake watermarks")

    # ========================================================================
    # STEP 5: Store metadata and create jobs
    # ========================================================================

    print(f"[FAKE 4090] Creating segment jobs...")
    self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Creating jobs'})

    redis_client = celery.backend.client

    metadata = {
        'video_id': video_id,
        'base_name': base_name,
        'video_path': str(video_path),
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': frames_extracted,
        'segments': segments,
        'frames_dir': frames_dir,
        'masks_dir': masks_dir,
        'api_base': api_base,
        'created_at': time.time(),
        'fake': True  # Mark as fake detection
    }

    redis_client.setex(
        f'video_metadata:{video_id}',
        3600,
        json.dumps(metadata)
    )

    # Create segment jobs - upload to B2 first, then create job with cloudflare URL
    job_ids = []

    for seg_idx, (start_frame, end_frame) in enumerate(segments):
        # Upload segment frames+masks to B2
        print(f"\n[FAKE 4090] Uploading segment {seg_idx} to B2...")
        cloudflare_url = upload_segment_to_b2(
            video_id, seg_idx, frames_dir, masks_dir, start_frame, end_frame
        )
        print(f"[FAKE 4090] Segment {seg_idx} available at: {cloudflare_url}")

        # Create job with cloudflare URL
        result = celery.send_task(
            'worker.process_segment',
            kwargs={
                'video_id': video_id,
                'segment_index': seg_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'cloudflare_url': cloudflare_url  # 3070 will download from here
            },
            queue='processing'
        )

        job_ids.append(result.id)
        print(f"[FAKE 4090] Created job {result.id} for segment {seg_idx}: frames {start_frame}-{end_frame}")

    # ========================================================================
    # DONE
    # ========================================================================

    self.update_state(state='PROCESSING', meta={'progress': 100, 'status': 'Detection complete'})

    result = {
        'status': 'success',
        'video_id': video_id,
        'total_frames': frames_extracted,
        'segments': segments,
        'job_ids': job_ids,
        'frames_with_watermark': frames_with_watermark,
        'fake': True
    }

    print(f"\n[FAKE 4090] Detection complete!")
    print(f"   Video ID: {video_id}")
    print(f"   Segments: {len(segments)}")
    print(f"   Jobs created: {len(job_ids)}")

    return result


@celery.task(bind=True, name='detection.check_status')
def check_segment_status(self, video_id):
    """Check status of segment jobs"""
    redis_client = celery.backend.client

    metadata_json = redis_client.get(f'video_metadata:{video_id}')
    if not metadata_json:
        return {'status': 'not_found', 'video_id': video_id}

    metadata = json.loads(metadata_json)
    segments = metadata['segments']

    completed = 0
    in_progress = 0
    pending = 0
    failed = 0

    for seg_idx in range(len(segments)):
        status_key = f'segment_status:{video_id}:{seg_idx}'
        status = redis_client.get(status_key)

        if status:
            status = status.decode() if isinstance(status, bytes) else status
            if status == 'completed':
                completed += 1
            elif status == 'processing':
                in_progress += 1
            elif status == 'failed':
                failed += 1
            else:
                pending += 1
        else:
            pending += 1

    return {
        'status': 'checking',
        'video_id': video_id,
        'total_segments': len(segments),
        'completed': completed,
        'in_progress': in_progress,
        'pending': pending,
        'failed': failed,
        'all_done': completed == len(segments)
    }


@worker_process_init.connect
def init_worker(**kwargs):
    print("[FAKE 4090 WORKER] Initialized - NO GPU REQUIRED")
    print("[FAKE 4090 WORKER] Ready to receive detection tasks!")


if __name__ == '__main__':
    print("Run with: celery -A celery_4090_fake worker -Q detection -c 1")
