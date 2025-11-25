"""
RTX 4090 Detection Server - Celery Worker for Watermark Detection

This runs on the cloud 4090 and handles:
1. Video upload/download
2. YOLO automatic detection OR SAM2 interactive detection
3. Creating segment jobs for 3070 workers to process

Architecture:
- 4090 (this): Detection + segment job creation
- 3070s: Process segment jobs from queue (ProPainter inpainting)
"""

import os
import sys
import json
import time
import uuid
import shutil
import hashlib
import zipfile
import numpy as np
from pathlib import Path

# Add python_packages to path
sys.path.insert(0, 'python_packages')

import cv2
from celery import Celery
from b2sdk.v2 import B2Api, InMemoryAccountInfo

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

print(f"[4090 DETECTION] Celery Broker = {broker}")

# Directory configuration
UPLOAD_DIR = os.path.join(SCRIPT_DIR, 'uploads')
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
MASK_DIR = os.path.join(SCRIPT_DIR, 'masks')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ============================================================================
# B2 + CLOUDFLARE CONFIGURATION
# ============================================================================

B2_KEY_ID = '00539db5c1104b50000000001'
B2_APPLICATION_KEY = 'K005VEORbg6RcsRad3jZPr9n4Fp7jWU'
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

celery = Celery('detection_4090', broker=broker, backend=backend)

celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    worker_prefetch_multiplier=1,  # Only 1 detection task at a time
    result_expires=3600,
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
)

# ============================================================================
# DETECTOR INITIALIZATION
# ============================================================================

# Global detector (initialized once per worker)
_yolo_detector = None

def get_yolo_detector():
    """Get or initialize YOLO detector (singleton)"""
    global _yolo_detector
    if _yolo_detector is None:
        from yolo_detector import YOLOWatermarkDetector
        print("[4090] Initializing YOLO detector...")
        _yolo_detector = YOLOWatermarkDetector()

        # Warmup
        print("[4090] Warming up YOLO...")
        dummy = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(64)]
        _ = _yolo_detector.detect_batch(dummy, confidence_threshold=0.15, batch_size=64)
        print("[4090] YOLO ready!")

    return _yolo_detector

# ============================================================================
# CELERY TASKS
# ============================================================================

@celery.task(bind=True, name='detection.detect_video')
def detect_video_task(self, video_path, mode='yolo', click_points=None, api_base=None):
    """
    Main detection task - runs on 4090

    Args:
        video_path: Path to video file
        mode: 'yolo' for automatic detection, 'sam2' for interactive
        click_points: List of click points for SAM2 mode
        api_base: Base URL for API (for workers to download)

    Returns:
        Dict with video_id, segments, and job IDs
    """
    print(f"\n{'='*60}")
    print(f"[4090 DETECTION] Processing: {video_path}")
    print(f"[4090 DETECTION] Mode: {mode}")
    print(f"{'='*60}\n")

    video_id = uuid.uuid4().hex[:8]
    base_name = Path(video_path).stem

    # Update status
    self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Loading video'})

    # ========================================================================
    # STEP 1: Load video and extract frames
    # ========================================================================

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise Exception(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    print(f"[4090] Video: {width}x{height} @ {fps} fps ({total_frames} frames)")

    # Load all frames to memory for batch processing
    print(f"[4090] Loading {total_frames} frames to memory...")
    self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Loading frames'})

    decode_start = time.time()
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)

    cap.release()
    frames_loaded = len(all_frames)
    decode_time = time.time() - decode_start
    print(f"[4090] Loaded {frames_loaded} frames: {decode_time:.3f}s")

    # ========================================================================
    # STEP 2: Run detection (YOLO or SAM2)
    # ========================================================================

    if mode == 'yolo':
        # YOLO automatic detection
        print(f"[4090] Running YOLO batch detection...")
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'YOLO detection'})

        detector = get_yolo_detector()

        batch_start = time.time()
        all_detections = detector.detect_batch(
            all_frames,
            confidence_threshold=0.15,
            padding=0,
            batch_size=64
        )
        batch_time = time.time() - batch_start
        ms_per_frame = (batch_time / max(frames_loaded, 1)) * 1000
        print(f"[4090] YOLO detection: {batch_time:.3f}s ({ms_per_frame:.2f}ms/frame)")

        # Create masks for each frame
        print(f"[4090] Creating masks...")
        self.update_state(state='PROCESSING', meta={'progress': 40, 'status': 'Creating masks'})

        masks_dir = os.path.join(MASK_DIR, f"{base_name}_{video_id}")
        os.makedirs(masks_dir, exist_ok=True)

        frames_with_watermark = 0
        for frame_idx, (frame, detections) in enumerate(zip(all_frames, all_detections)):
            if detections:
                mask = detector.create_mask(frame, detections, feather_pixels=21)
                frames_with_watermark += 1
            else:
                mask = np.zeros((height, width), dtype=np.uint8)

            cv2.imwrite(os.path.join(masks_dir, f"{frame_idx:04d}.png"), mask)

        print(f"[4090] {frames_with_watermark}/{frames_loaded} frames have watermarks")

    elif mode == 'sam2':
        # SAM2 interactive detection
        if not click_points:
            raise ValueError("SAM2 mode requires click_points")

        print(f"[4090] Running SAM2 with {len(click_points)} click points...")
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'SAM2 segmentation'})

        # TODO: Implement SAM2 detection here
        # For now, this is a placeholder
        raise NotImplementedError("SAM2 mode not yet implemented in celery_4090_detection.py")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ========================================================================
    # STEP 3: Segment video and create jobs
    # ========================================================================

    print(f"[4090] Segmenting video for parallel processing...")
    self.update_state(state='PROCESSING', meta={'progress': 60, 'status': 'Creating segments'})

    # Segment the video based on watermark presence
    # Each segment is a contiguous range of frames
    segments = []
    current_start = None

    for frame_idx, detections in enumerate(all_detections):
        has_watermark = bool(detections)

        if has_watermark:
            if current_start is None:
                current_start = frame_idx
        else:
            if current_start is not None:
                # End of watermark segment
                segments.append((current_start, frame_idx - 1))
                current_start = None

    # Don't forget last segment
    if current_start is not None:
        segments.append((current_start, frames_loaded - 1))

    # Merge nearby segments (within 30 frames)
    if segments:
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_end = merged[-1][1]
            if start - prev_end <= 30:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        segments = merged

    print(f"[4090] Created {len(segments)} segments: {segments}")

    # ========================================================================
    # STEP 4: Save frames and push segment jobs to queue
    # ========================================================================

    print(f"[4090] Saving frames and creating jobs...")
    self.update_state(state='PROCESSING', meta={'progress': 70, 'status': 'Creating jobs'})

    # Save original frames for workers to download
    frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_originals")
    os.makedirs(frames_dir, exist_ok=True)

    for idx, frame in enumerate(all_frames):
        cv2.imwrite(os.path.join(frames_dir, f"{idx:04d}.png"), frame)

    # Store metadata in Redis
    redis_client = celery.backend.client

    metadata = {
        'video_id': video_id,
        'base_name': base_name,
        'video_path': str(video_path),
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': frames_loaded,
        'segments': segments,
        'frames_dir': frames_dir,
        'masks_dir': masks_dir,
        'api_base': api_base,
        'created_at': time.time()
    }

    redis_client.setex(
        f'video_metadata:{video_id}',
        3600,  # 1 hour expiry
        json.dumps(metadata)
    )

    # Create segment processing jobs for 3070 workers
    # Upload each segment to B2 first so remote workers can download
    job_ids = []

    for seg_idx, (start_frame, end_frame) in enumerate(segments):
        # Upload segment frames+masks to B2
        print(f"\n[4090] Uploading segment {seg_idx} to B2...")
        cloudflare_url = upload_segment_to_b2(
            video_id, seg_idx, frames_dir, masks_dir, start_frame, end_frame
        )
        print(f"[4090] Segment {seg_idx} available at: {cloudflare_url}")

        # Push to segment processing queue with cloudflare URL
        result = celery.send_task(
            'worker.process_segment',
            kwargs={
                'video_id': video_id,
                'segment_index': seg_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'cloudflare_url': cloudflare_url  # Worker downloads from here
            },
            queue='processing'
        )

        job_ids.append(result.id)
        print(f"[4090] Created job {result.id} for segment {seg_idx}: frames {start_frame}-{end_frame}")

    # ========================================================================
    # DONE
    # ========================================================================

    self.update_state(state='PROCESSING', meta={'progress': 100, 'status': 'Detection complete'})

    result = {
        'status': 'success',
        'video_id': video_id,
        'total_frames': frames_loaded,
        'segments': segments,
        'job_ids': job_ids,
        'frames_with_watermark': frames_with_watermark,
        'metadata': metadata
    }

    print(f"\n[4090] Detection complete!")
    print(f"   Video ID: {video_id}")
    print(f"   Segments: {len(segments)}")
    print(f"   Jobs created: {len(job_ids)}")

    return result


@celery.task(bind=True, name='detection.check_status')
def check_segment_status(self, video_id):
    """
    Check status of all segment processing jobs for a video
    """
    redis_client = celery.backend.client

    # Get metadata
    metadata_json = redis_client.get(f'video_metadata:{video_id}')
    if not metadata_json:
        return {'status': 'not_found', 'video_id': video_id}

    metadata = json.loads(metadata_json)
    segments = metadata['segments']

    # Check each segment status
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


# ============================================================================
# WORKER STARTUP
# ============================================================================

@celery.signals.worker_process_init.connect
def init_worker(**kwargs):
    """Initialize detector when worker starts"""
    print("[4090 WORKER] Initializing detection worker...")
    get_yolo_detector()
    print("[4090 WORKER] Ready to receive detection tasks!")


if __name__ == '__main__':
    print("Run with: celery -A celery_4090_detection worker -Q detection -c 1")
