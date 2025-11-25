"""
FAKE RTX 3070 Processing Worker - For Testing Without GPU

This simulates the 3070 processing worker without running ProPainter.
Copies frames instead of inpainting for testing job distribution.
"""

import os
import sys
import json
import time
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, 'python_packages')

import cv2
from celery import Celery
from celery.signals import worker_process_init

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

print(f"[FAKE 3070] Celery Broker = {broker}")

# Directories
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================================
# CELERY SETUP
# ============================================================================

celery = Celery('fake_worker_3070', broker=broker, backend=backend)

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
)

# ============================================================================
# FAKE PROCESSING TASK
# ============================================================================

@celery.task(bind=True, name='worker.process_segment')
def process_segment_task(self, video_id, segment_index, start_frame, end_frame):
    """
    FAKE segment processing - copies frames instead of ProPainter
    """
    print(f"\n{'='*60}")
    print(f"[FAKE 3070] Processing segment {segment_index}")
    print(f"[FAKE 3070] Video: {video_id}, Frames: {start_frame}-{end_frame}")
    print(f"{'='*60}\n")

    # Update status in Redis
    redis_client = celery.backend.client
    status_key = f'segment_status:{video_id}:{segment_index}'
    redis_client.set(status_key, 'processing')

    try:
        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Loading metadata'
        })

        # ====================================================================
        # STEP 1: Get metadata
        # ====================================================================

        metadata_json = redis_client.get(f'video_metadata:{video_id}')
        if not metadata_json:
            raise Exception(f"Video metadata not found: {video_id}")

        metadata = json.loads(metadata_json)

        frames_dir = metadata['frames_dir']
        masks_dir = metadata['masks_dir']
        width = metadata['width']
        height = metadata['height']

        # Convert Windows paths to Linux paths for WSL
        import platform
        if platform.system() == 'Linux':
            if frames_dir.startswith('D:\\'):
                frames_dir = '/mnt/d/' + frames_dir[3:].replace('\\', '/')
            elif frames_dir.startswith('D:/'):
                frames_dir = '/mnt/d/' + frames_dir[3:]
            if masks_dir.startswith('D:\\'):
                masks_dir = '/mnt/d/' + masks_dir[3:].replace('\\', '/')
            elif masks_dir.startswith('D:/'):
                masks_dir = '/mnt/d/' + masks_dir[3:]

        print(f"[FAKE 3070] Metadata loaded: {width}x{height}")

        # ====================================================================
        # STEP 2: Prepare directories
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Preparing'
        })

        work_dir = os.path.join(TEMP_DIR, f"{video_id}_seg{segment_index}_fake")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)

        # ====================================================================
        # STEP 3: FAKE processing - copy frames with green tint
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Fake processing'
        })

        num_frames = end_frame - start_frame + 1
        print(f"[FAKE 3070] FAKE processing {num_frames} frames (copying with tint)...")

        process_start = time.time()
        output_frames = []

        for frame_idx in range(start_frame, end_frame + 1):
            src_path = os.path.join(frames_dir, f"{frame_idx:04d}.png")
            out_idx = frame_idx - start_frame

            if os.path.exists(src_path):
                # Read frame
                frame = cv2.imread(src_path)

                # Add green tint to show it was "processed"
                green_overlay = np.zeros_like(frame)
                green_overlay[:, :, 1] = 30  # Add green
                frame = cv2.addWeighted(frame, 0.9, green_overlay, 0.1, 0)

                # Add text showing it's fake processed
                cv2.putText(
                    frame,
                    f"FAKE SEG {segment_index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Save
                out_path = os.path.join(work_dir, f"{out_idx:04d}.png")
                cv2.imwrite(out_path, frame)
                output_frames.append(f"{out_idx:04d}.png")
            else:
                print(f"[FAKE 3070] WARNING: Frame not found: {src_path}")

            # Simulate processing time (10ms per frame)
            time.sleep(0.01)

        process_time = time.time() - process_start
        ms_per_frame = (process_time / num_frames) * 1000
        print(f"[FAKE 3070] Fake processing: {process_time:.2f}s ({ms_per_frame:.2f}ms/frame)")

        # ====================================================================
        # STEP 4: Store results
        # ====================================================================

        self.update_state(state='PROCESSING', meta={
            'segment': segment_index,
            'status': 'Storing results'
        })

        result_dir = os.path.join(RESULT_DIR, f"{video_id}_segments", f"segment_{segment_index}")
        os.makedirs(result_dir, exist_ok=True)

        for frame_file in output_frames:
            src = os.path.join(work_dir, frame_file)
            dst = os.path.join(result_dir, frame_file)
            shutil.copy(src, dst)

        # Mark complete
        redis_client.set(status_key, 'completed')

        # Publish completion
        completion_data = {
            'video_id': video_id,
            'segment_index': segment_index,
            'output_path': result_dir,
            'num_frames': len(output_frames),
            'process_time': process_time,
            'fake': True
        }
        redis_client.publish(f'segment_complete:{video_id}', json.dumps(completion_data))

        # Cleanup
        shutil.rmtree(work_dir)

        result = {
            'status': 'success',
            'video_id': video_id,
            'segment_index': segment_index,
            'output_path': result_dir,
            'num_frames': len(output_frames),
            'process_time': process_time,
            'ms_per_frame': ms_per_frame,
            'fake': True
        }

        print(f"\n[FAKE 3070] Segment {segment_index} complete!")
        print(f"   Output: {result_dir}")
        print(f"   Frames: {len(output_frames)}")

        return result

    except Exception as e:
        redis_client.set(status_key, 'failed')
        print(f"\n[FAKE 3070 ERROR] Segment {segment_index} failed: {e}")

        return {
            'status': 'failed',
            'video_id': video_id,
            'segment_index': segment_index,
            'error': str(e)
        }


@celery.task(bind=True, name='worker.health_check')
def health_check(self):
    """Health check"""
    return {
        'status': 'healthy',
        'fake': True,
        'timestamp': time.time()
    }


@worker_process_init.connect
def init_worker(**kwargs):
    print("[FAKE 3070 WORKER] Initialized - NO GPU REQUIRED")
    print("[FAKE 3070 WORKER] Ready to process segments!")


if __name__ == '__main__':
    print("Run with: celery -A celery_3070_fake worker -Q processing -c 1")
