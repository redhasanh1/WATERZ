"""
SAM2 Watermark Removal Server - Parallel Production System
Isolated from YOLO production (server_production.py)

This server handles SAM2-based watermark removal using pre-generated masks.
"""

import os
import sys

# [CRITICAL] Enable NeuFlow v2 TensorRT for optical flow (10-70x faster than RAFT)
# Must be set BEFORE importing watermark.py / ProPainter pipeline
os.environ['USE_NEUFLOW'] = '1'
import glob
import json
import threading
import time
import uuid
from typing import List, Optional, Tuple
from contextlib import nullcontext
from flask import Flask, request, jsonify
from celery import Celery
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import torch

# Segment detection for parallel processing
from segment_detector import detect_segments_from_masks, merge_adjacent_segments, detect_segments, detect_segments_motion_based
from crop_utils import calculate_crop_region

# Number of parallel segment workers
SEGMENT_WORKERS = int(os.getenv('SEGMENT_WORKERS', '2'))
# Parallel segment execution (inside a single video task)
SAM2_PARALLEL_SEGMENTS = os.getenv('SAM2_PARALLEL_SEGMENTS', '1').lower() in ('1', 'true', 'yes', 'on')

# Temporal padding (in full-FPS frames) around each detected segment
# Helps prevent reappearance at segment boundaries by giving ProPainter context
SEGMENT_TEMPORAL_PAD = int(os.getenv('SEGMENT_TEMPORAL_PAD', '5'))

# Inpainting mask dilation (higher = stronger removal, more context fill)
SAM2_MASK_DILATION = int(os.getenv('SAM2_MASK_DILATION', '4'))
# Strict mode: use single full-video inpainting (no segment split)
SAM2_USE_SEGMENTS = os.getenv('SAM2_USE_SEGMENTS', '0') in ('1', 'true', 'yes', 'on')

# Segment detection sensitivity
# Higher tolerance/gap = less sensitive (fewer segments)
SEGMENT_POS_TOLERANCE = int(os.getenv('SEGMENT_POS_TOLERANCE', '50'))  # match commit defaults
# 10fps domain (legacy fallback)
SEGMENT_MIN_LEN_10FPS = int(os.getenv('SEGMENT_MIN_LEN_10FPS', '10'))
SEGMENT_MERGE_GAP_10FPS = int(os.getenv('SEGMENT_MERGE_GAP_10FPS', '6'))
# Full-FPS domain (preferred)
SEGMENT_MIN_LEN_FULL = int(os.getenv('SEGMENT_MIN_LEN_FULL', '15'))
SEGMENT_MERGE_GAP_FULL = int(os.getenv('SEGMENT_MERGE_GAP_FULL', '60'))
# Motion-based detection (better for fast-moving objects)
# SEGMENT_MOTION_THRESHOLD: max px movement between frames before new segment (default 20)
SEGMENT_MOTION_THRESHOLD = int(os.getenv('SEGMENT_MOTION_THRESHOLD', '20'))
# Use motion-based detection instead of average-based (1=motion, 0=average)
SEGMENT_USE_MOTION_DETECTION = os.getenv('SEGMENT_USE_MOTION_DETECTION', '1').lower() in ('1', 'true', 'yes', 'on')
# Max frames per segment (longer segments get split to prevent OOM)
MAX_SEGMENT_FRAMES = int(os.getenv('MAX_SEGMENT_FRAMES', '300'))
# Max pixels (width*height) for segment's union bbox (0=unlimited, 400000=~630x630 recommended)
MAX_SEGMENT_PIXELS = int(os.getenv('MAX_SEGMENT_PIXELS', '400000'))
# Max pixels AFTER padding - triggers segment splitting if exceeded (~775x775 = 600k)
MAX_CROP_PIXELS = int(os.getenv('MAX_CROP_PIXELS', '600000'))
# Detection mode: 'full' (preferred, uses in-memory masks) or '10fps' (dir-based)
SAM2_SEGMENT_DETECTION_MODE = os.getenv('SAM2_SEGMENT_DETECTION_MODE', 'full').strip().lower()

# Tail mask fill (replicate last non-empty mask to the end to avoid missed final seconds)
TAIL_MASK_FILL = os.getenv('TAIL_MASK_FILL', '1').lower() in ('1', 'true', 'yes', 'on')
try:
    TAIL_MASK_MAX_SECONDS = float(os.getenv('TAIL_MASK_MAX_SECONDS', '2.0'))
except Exception:
    TAIL_MASK_MAX_SECONDS = 2.0

# Fill short internal mask gaps (replicate nearest non-empty mask across brief gaps)
FILL_MASK_GAPS = os.getenv('FILL_MASK_GAPS', '0').lower() in ('1', 'true', 'yes', 'on')
try:
    MASK_GAP_MAX_SECONDS = float(os.getenv('MASK_GAP_MAX_SECONDS', '1.0'))
except Exception:
    MASK_GAP_MAX_SECONDS = 1.0

# Per-segment crop padding ratio (increase to avoid missing moving object near crop edges)
try:
    SEGMENT_CROP_PAD_RATIO = float(os.getenv('SEGMENT_CROP_PAD_RATIO', '0.25'))
except Exception:
    SEGMENT_CROP_PAD_RATIO = 0.25


def split_segment_by_pixels(start, end, detections, max_pixels, padding_ratio=0.25, frame_width=1920, frame_height=1080):
    """
    Recursively split segment until each piece has crop < max_pixels AFTER padding.
    Uses calculate_crop_region for EXACT crop size (includes min_size, clamping).
    """
    bboxes = [detections[f] for f in range(start, end)
              if f < len(detections) and detections[f] is not None]
    if not bboxes:
        return [(start, end, (0, 0, 0, 0))]

    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    bbox = (x1, y1, x2, y2)

    # Use calculate_crop_region to get EXACT crop size (includes min_size, clamping)
    _, _, crop_w, crop_h = calculate_crop_region(
        bbox, frame_width, frame_height,
        padding_ratio=padding_ratio, min_size=128
    )
    crop_pixels = crop_w * crop_h

    # Stop splitting if under limit OR segment is too short (min 30 frames)
    if crop_pixels <= max_pixels or end - start <= 30:
        return [(start, end, bbox)]

    # Binary split
    mid = (start + end) // 2
    left = split_segment_by_pixels(start, mid, detections, max_pixels, padding_ratio, frame_width, frame_height)
    right = split_segment_by_pixels(mid, end, detections, max_pixels, padding_ratio, frame_width, frame_height)
    return left + right


class VRAMCompressor:
    """
    YUV420 compression for VRAM storage - 8:1 ratio, 43.5dB PSNR.
    Stores frames in GPU memory with minimal quality loss.
    """

    @staticmethod
    def compress_frame(frame_bgr):
        """Compress single BGR frame to YUV420 in VRAM
        Args: frame_bgr - numpy array [H, W, 3] uint8
        Returns: dict with y, u, v tensors on GPU (8:1 smaller)
        """
        import torch
        import torch.nn.functional as F

        frame_gpu = torch.from_numpy(frame_bgr).cuda().float()
        b, g, r = frame_gpu[:, :, 0], frame_gpu[:, :, 1], frame_gpu[:, :, 2]

        # RGB to YUV (BT.601)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b + 128
        v = 0.615 * r - 0.515 * g - 0.100 * b + 128

        # 4:2:0 subsampling - Y full res, U/V quarter res
        y_full = y.byte()
        u_sub = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), 2).squeeze().byte()
        v_sub = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), 2).squeeze().byte()

        return {'y': y_full, 'u': u_sub, 'v': v_sub, 'shape': frame_bgr.shape}

    @staticmethod
    def decompress_frame(compressed):
        """Decompress YUV420 back to BGR frame
        Returns: numpy array [H, W, 3] uint8
        """
        import torch
        import torch.nn.functional as F

        y = compressed['y'].float()
        u = F.interpolate(compressed['u'].unsqueeze(0).unsqueeze(0).float(), scale_factor=2, mode='bilinear').squeeze()
        v = F.interpolate(compressed['v'].unsqueeze(0).unsqueeze(0).float(), scale_factor=2, mode='bilinear').squeeze()

        u_adj, v_adj = u - 128, v - 128
        r = (y + 1.140 * v_adj).clamp(0, 255)
        g = (y - 0.395 * u_adj - 0.581 * v_adj).clamp(0, 255)
        b = (y + 2.032 * u_adj).clamp(0, 255)

        return torch.stack([b, g, r], dim=-1).byte().cpu().numpy()


# [YOLO] EXTREME SPEED: Global in-memory frame/mask cache
# Shared across all threads in Celery worker (threads pool)
# Stores frames/masks in RAM for instant access (no Redis, no disk!)
FRAME_CACHE = {}
FRAME_CACHE_LOCK = threading.Lock()

# [YOLO] Global YOLO detector (TensorRT)
detector = None


def get_detector():
    """
    Lazy load the YOLO detector (TensorRT batch engine).
    Uses Windows TensorRT engine for maximum speed.
    """
    global detector

    if detector is None:
        print("=" * 60)
        print("Loading YOLO detector (Windows TensorRT)...")
        print("=" * 60)
        from yolo_detector import YOLOWatermarkDetector
        # Force TensorRT-only mode (no fallback to .pt)
        # Will fail if engine not found - ensures maximum speed!
        detector = YOLOWatermarkDetector(require_tensorrt=True)

        # WARMUP: Initialize TensorRT context (eliminates cold start overhead!)
        print("[WARMUP] Running warmup inference to initialize TensorRT context...")
        dummy_batch = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(64)]
        _ = detector.detect_batch(dummy_batch, confidence_threshold=0.15, batch_size=64)
        print("[OK] YOLO warmed up! TensorRT context ready for max speed.")

        print("=" * 60)
        print("[OK] YOLO detector ready!")
        print("=" * 60)

    return detector


# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
PROPAINTER_DIR = os.path.join(BASE_DIR, 'faster-propainter-main')

# Add ProPainter to path
sys.path.insert(0, PROPAINTER_DIR)

# Create directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Load Redis URL
REDIS_URL = os.getenv('REDIS_URL', 'redis://:watermarkz_secure_2024@localhost:6379/0')
if os.path.exists('redis_url.txt'):
    with open('redis_url.txt', 'r') as f:
        REDIS_URL = f.read().strip()

# Configure Celery (new format)
celery = Celery(app.name, broker=REDIS_URL, backend=REDIS_URL)
celery.conf.task_track_started = True
try:
    # Allow tuning via environment; CLI flags still take precedence when provided.
    _pool = os.getenv('CELERY_POOL', None)
    _conc = os.getenv('CELERY_CONCURRENCY', os.getenv('WORKER_CONCURRENCY', None))
    if _pool:
        celery.conf.worker_pool = _pool
    if _conc:
        celery.conf.worker_concurrency = int(_conc)
except Exception:
    pass

# Get FFmpeg/FFprobe executables
def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths - check multiple locations"""
    # Check local ffmpeg folder first
    ffmpeg_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffmpeg.exe')
    ffprobe_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffprobe.exe')

    # Check static_ffmpeg package location
    if not os.path.exists(ffmpeg_exe):
        try:
            import static_ffmpeg
            static_path = os.path.dirname(static_ffmpeg.__file__)
            static_ffmpeg_exe = os.path.join(static_path, 'bin', 'win32', 'ffmpeg.exe')
            static_ffprobe_exe = os.path.join(static_path, 'bin', 'win32', 'ffprobe.exe')
            if os.path.exists(static_ffmpeg_exe):
                ffmpeg_exe = static_ffmpeg_exe
                ffprobe_exe = static_ffprobe_exe
                print(f"[FFmpeg] Using static_ffmpeg: {ffmpeg_exe}")
        except ImportError:
            pass

    # Final fallback to system PATH
    if not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = 'ffmpeg'
    if not os.path.exists(ffprobe_exe):
        ffprobe_exe = 'ffprobe'

    return ffmpeg_exe, ffprobe_exe

FFMPEG_EXE, FFPROBE_EXE = get_ffmpeg_executables()


def read_video_frames_range(video_path: str, start_frame: int, end_frame: int,
                            crop_region: Optional[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
    """
    Read frames [start_frame, end_frame) from video, optionally crop on-the-fly.
    This prevents loading entire video into memory - only loads the requested range.

    Args:
        video_path: Path to the video file
        start_frame: First frame to read (0-indexed)
        end_frame: One past the last frame to read
        crop_region: Optional (x, y, w, h) tuple to crop each frame

    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if crop_region:
            cx, cy, cw, ch = crop_region
            frame = frame[cy:cy+ch, cx:cx+cw]
        frames.append(frame)
    cap.release()
    return frames


# Configurable tracking FPS (default 15fps - good balance for fast objects)
SAM2_TRACK_FPS = int(os.getenv('SAM2_TRACK_FPS', '15'))

def convert_to_tracking_fps(video_path: str, output_path: str = None, target_fps: int = None) -> str:
    """
    Convert video to target FPS using FFmpeg for SAM2 tracking.
    Returns path to the converted video.

    Args:
        video_path: Path to input video
        output_path: Optional output path (auto-generated if None)
        target_fps: Target FPS (defaults to SAM2_TRACK_FPS env var)

    Returns:
        Path to converted video
    """
    if target_fps is None:
        target_fps = SAM2_TRACK_FPS

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_{target_fps}fps{ext}"

    # Try GPU acceleration first, fall back to CPU
    # Using scale filter to maintain quality while reducing framerate
    cmd = [
        str(FFMPEG_EXE),
        '-y',
        '-i', video_path,
        '-vf', f'fps={target_fps}',
        '-c:v', 'libx264',  # CPU fallback (more compatible)
        '-preset', 'ultrafast',
        '-crf', '18',
        '-an',  # No audio for processing
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg {target_fps}fps conversion failed: {result.stderr[:500]}")
        raise RuntimeError(f"{target_fps}fps conversion failed: {result.stderr[:200]}")

    print(f"[OK] Converted to {target_fps}fps: {output_path}")
    return output_path


# Check ProPainter assets
def _check_propainter_assets():
    """Check if ProPainter model checkpoints exist"""
    required_files = [
        os.path.join(PROPAINTER_DIR, 'weights', 'ProPainter.pth'),
        os.path.join(PROPAINTER_DIR, 'weights', 'recurrent_flow_completion.pth')
    ]
    return all(os.path.exists(f) for f in required_files)

# Get ProPainter pipeline (cached at worker startup)
_propainter_pipeline = None

def get_propainter_pipeline():
    """Get cached ProPainter pipeline"""
    global _propainter_pipeline
    if _propainter_pipeline is None:
        from watermark import pipeline as faster_propainter_pipeline
        _propainter_pipeline = faster_propainter_pipeline
    return _propainter_pipeline

def clear_gpu_memory():
    """Clear GPU memory between segments"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

#============================================================================
# YOLO DETECTOR (for YOLO mode - separate from SAM2)
#============================================================================

# Global YOLO detector (lazy loaded)
_yolo_detector = None

def get_yolo_detector():
    """
    Lazy load the YOLO detector (TensorRT if available).
    Used by YOLO mode Celery workers for fast watermark detection.
    """
    global _yolo_detector

    if _yolo_detector is None:
        print("=" * 60)
        print("Loading YOLO detector...")
        print("=" * 60)
        from yolo_detector import YOLOWatermarkDetector
        import numpy as np
        # Force TensorRT-only mode (no fallback to .pt) for max speed
        require_trt = os.getenv('YOLO_REQUIRE_TENSORRT', '0') == '1'
        _yolo_detector = YOLOWatermarkDetector(require_tensorrt=require_trt)

        # WARMUP: Initialize TensorRT context (eliminates cold start overhead!)
        print("[WARMUP] Running warmup inference to initialize TensorRT context...")
        dummy_batch = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(64)]
        _ = _yolo_detector.detect_batch(dummy_batch, confidence_threshold=0.15, batch_size=64)
        print("[OK] YOLO warmed up! TensorRT context ready for max speed.")

        print("=" * 60)
        print("[OK] YOLO detector ready!")
        print("=" * 60)

    return _yolo_detector

#============================================================================
# BACKGROUND ENCODER (for YOLO mode - encodes segments as they complete)
# Workers signal when segments ready → Background thread encodes immediately
# Encoding happens continuously as segments complete (not all at once at end)
#============================================================================

_yolo_encoder_thread = None

def start_yolo_background_encoder(**kwargs):
    """
    Start background encoding thread when worker process initializes.
    This runs ONCE per worker process.
    """
    global _yolo_encoder_thread

    print("[YOLO BACKGROUND ENCODER INIT] Worker process starting, initializing background encoder...")

    if _yolo_encoder_thread is None or not _yolo_encoder_thread.is_alive():
        print("[YOLO BACKGROUND ENCODER] Starting real-time encoding thread...")
        _yolo_encoder_thread = threading.Thread(
            target=yolo_background_encoder_worker,
            daemon=True,  # Dies with worker process
            name="YoloBackgroundEncoder"
        )
        _yolo_encoder_thread.start()
        print("[YOLO BACKGROUND ENCODER] Thread active - will encode segments as they complete!")
    else:
        print("[YOLO BACKGROUND ENCODER] Thread already running")


def yolo_background_encoder_worker():
    """
    Background thread that encodes segments in real-time as they complete.
    Uses Redis pub/sub to receive segment completion notifications.
    PARALLEL ENCODING: Encodes multiple segments simultaneously using ThreadPoolExecutor.
    """
    import json
    import traceback
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    # Track videos being processed
    video_futures = {}  # {video_id: {seg_idx: future, ...}}
    video_metadata = {}  # {video_id: {'total_segments': N, 'redis_client': client}}

    while True:  # Auto-reconnect loop
        try:
            redis_client = celery.backend.client

            # Configure pubsub with NO timeout and keepalive
            pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
            pubsub.connection_pool.connection_kwargs['socket_timeout'] = None
            pubsub.connection_pool.connection_kwargs['socket_keepalive'] = True
            pubsub.connection_pool.connection_kwargs['socket_keepalive_options'] = {
                1: 60,   # TCP_KEEPIDLE
                2: 10,   # TCP_KEEPINTVL
                3: 6     # TCP_KEEPCNT
            }

            pubsub.subscribe('yolo_segment_ready')

            print("[YOLO BACKGROUND ENCODER] Listening for segment completion signals...")
            print("[YOLO BACKGROUND ENCODER] Socket keepalive enabled - NO timeout!")
            print("[YOLO BACKGROUND ENCODER] PARALLEL MODE: Up to 4 concurrent NVENC streams!")

            # Create thread pool for parallel encoding (4 matches SEGMENT_WORKERS)
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="YoloEncoderThread") as executor:
                for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            video_id = data['video_id']
                            seg_idx = data['seg_idx']
                            total_segments = data['total_segments']

                            # Initialize tracking for new video
                            if video_id not in video_futures:
                                video_futures[video_id] = {}
                                video_metadata[video_id] = {
                                    'total_segments': total_segments,
                                    'redis_client': redis_client
                                }

                            print(f"[YOLO BACKGROUND ENCODER] Segment {seg_idx+1}/{total_segments} ready for video {video_id} - submitting to parallel encoder!")

                            # Submit encoding to thread pool (NON-BLOCKING!)
                            future = executor.submit(yolo_encode_segment_background, redis_client, data)
                            video_futures[video_id][seg_idx] = future

                            # Check if all segments have been submitted for this video
                            if len(video_futures[video_id]) == total_segments:
                                print(f"[YOLO BACKGROUND ENCODER] All {total_segments} segments submitted for {video_id} - waiting for completion...")

                                # Wait for all encoding futures to complete
                                completed_count = 0
                                failed_segments = []

                                for seg_idx_done, future in video_futures[video_id].items():
                                    try:
                                        future.result()  # Block until this segment completes
                                        completed_count += 1
                                        print(f"[YOLO BACKGROUND ENCODER] Segment {seg_idx_done+1} encoded! Progress: {completed_count}/{total_segments}")
                                    except Exception as e:
                                        print(f"[YOLO BACKGROUND ENCODER ERROR] Segment {seg_idx_done+1} failed: {e}")
                                        traceback.print_exc()
                                        failed_segments.append(seg_idx_done)

                                # Finalize if all segments succeeded
                                if not failed_segments:
                                    print(f"[YOLO BACKGROUND ENCODER] All {total_segments} segments encoded for {video_id}! Triggering finalization...")
                                    yolo_trigger_finalization(redis_client, video_id, total_segments)
                                    print(f"[YOLO BACKGROUND ENCODER] Video {video_id} finalized!")
                                else:
                                    print(f"[YOLO BACKGROUND ENCODER ERROR] Video {video_id} had {len(failed_segments)} failed segments: {failed_segments}")

                                # Cleanup tracking
                                del video_futures[video_id]
                                del video_metadata[video_id]

                        except Exception as e:
                            print(f"[YOLO BACKGROUND ENCODER ERROR] Failed to process segment: {e}")
                            traceback.print_exc()

        except Exception as e:
            print(f"[YOLO BACKGROUND ENCODER] Connection lost: {e}")
            traceback.print_exc()
            print("[YOLO BACKGROUND ENCODER] Reconnecting in 2 seconds...")
            _time.sleep(2)
            # Loop continues - auto-reconnect!


def yolo_encode_segment_background(redis_client, data):
    """
    Encode a segment using GPU NVENC (same encoding as before, just in background).
    Called by background encoder thread continuously as segments complete.
    """
    import subprocess
    import time as _time

    video_id = data['video_id']
    seg_idx = data['seg_idx']
    total_segments = data['total_segments']

    # Get segment metadata from Redis
    segment_key = f"yolo_video:{video_id}:segment:{seg_idx}"
    segment_info_raw = redis_client.hgetall(segment_key)

    if not segment_info_raw:
        raise RuntimeError(f"Segment metadata not found in Redis: {segment_key}")

    # Decode bytes to strings (Redis returns bytes)
    segment_info = {
        k.decode() if isinstance(k, bytes) else k:
        v.decode() if isinstance(v, bytes) else v
        for k, v in segment_info_raw.items()
    }

    cleaned_dir = segment_info.get('cleaned_dir')
    fps = float(segment_info.get('fps', 30))
    frame_count = int(segment_info.get('frame_count', 0))
    base_name = segment_info.get('base_name', 'video')
    start_frame = int(segment_info.get('start_frame', 0))
    end_frame = int(segment_info.get('end_frame', frame_count - 1))

    if not cleaned_dir or not os.path.exists(cleaned_dir):
        raise RuntimeError(f"Cleaned frames directory not found: {cleaned_dir}")

    # Create output path
    seg_video_path = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_seg{seg_idx}.mp4")

    print(f"[YOLO ENCODER] Encoding segment {seg_idx}: frames {start_frame}-{end_frame} ({frame_count} frames) @ {fps} fps...")
    encode_start = _time.time()

    # Create file list for this segment's frames only
    file_list_path = os.path.join(TEMP_DIR, f"yolo_encode_seg{seg_idx}_{video_id}.txt")
    with open(file_list_path, 'w') as f:
        for global_idx in range(start_frame, end_frame + 1):
            frame_path = os.path.join(cleaned_dir, f"{global_idx:04d}.png")
            if os.path.exists(frame_path):
                abs_path = os.path.abspath(frame_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {1/fps}\n")
        # Last frame needs to be repeated for proper duration
        last_frame_path = os.path.join(cleaned_dir, f"{end_frame:04d}.png")
        if os.path.exists(last_frame_path):
            abs_path = os.path.abspath(last_frame_path).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    # Encode with NVENC using file list (concat demuxer)
    encode_cmd = [
        str(FFMPEG_EXE), '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',  # Balanced speed/quality
        '-b:v', '8M',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'main',
        seg_video_path
    ]

    try:
        result = subprocess.run(encode_cmd, capture_output=True, check=True, text=True, timeout=300)
        encode_duration = _time.time() - encode_start

        encoded_size_mb = os.path.getsize(seg_video_path) / (1024 * 1024)
        fps_actual = frame_count / encode_duration if encode_duration > 0 else 0

        print(f"[YOLO ENCODER] [OK] Encoded: {encoded_size_mb:.2f} MB in {encode_duration:.2f}s ({fps_actual:.1f} fps)")

        # Store encoded path in Redis
        redis_client.hset(segment_key, 'encoded_path', seg_video_path)
        redis_client.hset(segment_key, 'status', 'encoded')

        # Cleanup file list after successful encoding
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

    except subprocess.CalledProcessError as e:
        print(f"[YOLO ENCODER ERROR] Encoding failed for segment {seg_idx}!")
        print(f"   stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print(f"[YOLO ENCODER ERROR] Encoding timed out after 300s for segment {seg_idx}")
        raise


def yolo_trigger_finalization(redis_client, video_id, total_segments):
    """
    Concatenate all encoded segments and merge audio.
    Called automatically when all segments are encoded.
    """
    import subprocess
    import shutil

    print(f"\n[YOLO FINALIZE] Starting finalization for video {video_id}")

    # Collect all segment video paths from Redis (in order)
    segment_paths = []
    for seg_idx in range(total_segments):
        segment_key = f"yolo_video:{video_id}:segment:{seg_idx}"
        encoded_path_raw = redis_client.hget(segment_key, 'encoded_path')

        # Decode bytes to string
        encoded_path = encoded_path_raw.decode() if isinstance(encoded_path_raw, bytes) else encoded_path_raw

        if not encoded_path or not os.path.exists(encoded_path):
            raise RuntimeError(f"Missing encoded segment {seg_idx}: {encoded_path}")

        segment_paths.append(encoded_path)

    print(f"[YOLO FINALIZE] Found {len(segment_paths)} encoded segments")

    # Get video metadata (decode bytes)
    base_name_raw = redis_client.get(f"yolo_video:{video_id}:base_name")
    base_name = base_name_raw.decode() if isinstance(base_name_raw, bytes) else (base_name_raw or 'video')

    video_path_raw = redis_client.get(f"yolo_video:{video_id}:video_path")
    video_path = video_path_raw.decode() if isinstance(video_path_raw, bytes) else video_path_raw

    # Create concat list
    concat_list_path = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_yolo_concat.txt")
    with open(concat_list_path, 'w') as f:
        for seg_path in segment_paths:
            abs_path = os.path.abspath(seg_path).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    # Concatenate with copy codec (instant - no re-encoding)
    temp_processed = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_yolo_processed.mp4")

    concat_cmd = [
        str(FFMPEG_EXE), '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',  # No re-encoding - instant!
        temp_processed
    ]

    print(f"[YOLO FINALIZE] Concatenating segments with copy codec...")
    subprocess.run(concat_cmd, capture_output=True, check=True, text=True, timeout=60)
    concat_size_mb = os.path.getsize(temp_processed) / (1024 * 1024)
    print(f"[YOLO FINALIZE] Concatenated: {concat_size_mb:.2f} MB")

    # Merge audio from original
    final_output = os.path.join(RESULT_DIR, f"{base_name}_yolo_propainter.mp4")

    if video_path and os.path.exists(video_path):
        # Check if original has audio
        check_audio_cmd = [
            str(FFPROBE_EXE), '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
        has_audio = 'audio' in has_audio_check.stdout

        if has_audio:
            print(f"[YOLO FINALIZE] Merging audio from original...")
            merge_cmd = [
                str(FFMPEG_EXE), '-y',
                '-i', temp_processed,
                '-i', video_path,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'copy',
                '-shortest',
                final_output
            ]
            subprocess.run(merge_cmd, capture_output=True, check=True, text=True, timeout=300)
            if os.path.exists(temp_processed):
                os.remove(temp_processed)
            print(f"[YOLO FINALIZE] Audio merged")
        else:
            os.rename(temp_processed, final_output)
            print(f"[YOLO FINALIZE] No audio in original")
    else:
        os.rename(temp_processed, final_output)
        print(f"[YOLO FINALIZE] Using processed video only")

    # Cleanup
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    for seg_path in segment_paths:
        if os.path.exists(seg_path):
            os.remove(seg_path)

    # Cleanup shared frame directory after finalization
    segment_key = f"yolo_video:{video_id}:segment:0"
    shared_cleaned_dir_raw = redis_client.hget(segment_key, 'cleaned_dir')
    if shared_cleaned_dir_raw:
        shared_cleaned_dir = shared_cleaned_dir_raw.decode() if isinstance(shared_cleaned_dir_raw, bytes) else shared_cleaned_dir_raw
        if shared_cleaned_dir and os.path.exists(shared_cleaned_dir):
            print(f"[YOLO FINALIZE] Cleaning up shared frame buffer: {shared_cleaned_dir}")
            shutil.rmtree(shared_cleaned_dir, ignore_errors=True)

    final_size_mb = os.path.getsize(final_output) / (1024 * 1024)
    print(f"[YOLO FINALIZE] Final video ready: {final_output} ({final_size_mb:.2f} MB)")

    # Store final result in Redis
    redis_client.set(f"yolo_video:{video_id}:final_path", final_output)
    redis_client.set(f"yolo_video:{video_id}:status", "complete")

    # Update distributed tracking to mark all segments complete
    tracking_key = f"yolo_segments:{video_id}"
    total_segments_bytes = redis_client.get(f"{tracking_key}:total")
    if total_segments_bytes:
        redis_client.set(tracking_key, int(total_segments_bytes))
        print(f"[YOLO FINALIZE] Marked all {int(total_segments_bytes)} segments complete in Redis tracking")


# Register background encoder to start when Celery worker initializes
from celery.signals import worker_process_init
worker_process_init.connect(start_yolo_background_encoder)

#============================================================================
# CELERY TASK: SAM2 Interactive Mode - FULL FPS Pipeline
#
# Optimizations:
# - Use original FPS for SAM2 tracking (more accurate masks)
# - Run ProPainter with in-memory arrays (ZERO disk I/O)
# - Use NeuFlow TRT for optical flow (10-70x faster than RAFT)
#============================================================================

@celery.task(bind=True, name='watermark.process_sam2_interactive')
def process_sam2_interactive_task(self, video_path, video_id=None, points=None, video_width=None, video_height=None, frame_index=0, api_base=None):
    """
    SAM2 Interactive Mode - FULL FPS Pipeline:
    - Downloads video from remote
    - Generates masks at original FPS using SAM2Tracker (WSL2)
    - Runs ProPainter with in-memory arrays (zero disk I/O)
    - Returns processed video with audio

    Args:
        video_path (str): Path to input video (on remote server)
        video_id (str): Video ID for tracking
        points (list): User selection points
        video_width (int): Width of the video
        video_height (int): Height of the video
        frame_index (int): The frame index the user clicked on (at original FPS)
    """
    try:
        import shutil
        import json
        import numpy as np
        import cv2
        import requests
        from urllib.parse import urljoin
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Initializing SAM2 FULL-FPS Pipeline'})

        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing")

        # Generate video ID if not provided
        if not video_id:
            video_id = os.path.basename(video_path).split('.')[0][:8]

        # --- 1. Locate video (robust local/remote handling) ---
        from pathlib import PureWindowsPath

        # Extract filename from video_path (handles Windows paths)
        base_name = PureWindowsPath(video_path).name if '\\' in video_path else os.path.basename(video_path)
        UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
        local_video_path = os.path.join(UPLOAD_DIR, base_name)

        # Smart check: If file exists locally (same PC workers share uploads/), skip download!
        if os.path.exists(local_video_path):
            print(f"[SAM2] Video already exists locally: {local_video_path} (skip download)")
            video_path = local_video_path
        elif not os.path.exists(video_path):
            # File not found at provided path - try to download
            # 1) If provided path is an absolute URL, use it directly
            if isinstance(video_path, str) and video_path.lower().startswith(('http://', 'https://')):
                download_url = video_path
            else:
                # 2) Construct from API base or env TUNNEL_URL or optional file 'tunnel_url.txt'
                tunnel = api_base or os.getenv('TUNNEL_URL')
                if not tunnel:
                    for candidate in ('tunnel_url.txt', 'TUNNEL_URL.txt'):
                        fp = os.path.join(BASE_DIR, candidate)
                        if os.path.exists(fp):
                            try:
                                with open(fp, 'r') as fh:
                                    tunnel = fh.read().strip()
                                    if tunnel:
                                        print(f"[SAM2] Loaded TUNNEL_URL from {candidate}: {tunnel}")
                                        break
                            except Exception:
                                pass
                if not tunnel:
                    # 3) As a last resort, attempt to map '/data/uploads/<file>' → local uploads/<file>
                    #    If still missing, fail with actionable error
                    if os.path.exists(local_video_path):
                        video_path = local_video_path
                    else:
                        raise RuntimeError(f"Video not found locally and no TUNNEL_URL provided: {video_path}")
                else:
                    download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')

            if 'download_url' in locals():
                print(f"[SAM2] Downloading video: {download_url}")
                self.update_state(state='STARTED', meta={'progress': 1, 'status': 'Downloading video'})
                r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=120)
                r.raise_for_status()
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(local_video_path, 'wb') as f:
                    f.write(r.content)
                print(f"[SAM2] Video downloaded to: {local_video_path}")
                video_path = local_video_path
        # else: video_path exists, use it as-is (local file path provided directly)

        # --- 2. Get video metadata (FPS, frame count) ---
        self.update_state(state='PROCESSING', meta={'progress': 3, 'status': 'Analyzing video'})
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"[SAM2] Video: {width}x{height} @ {original_fps}fps, {total_frames_original} frames")

        # --- 3. Use original FPS for SAM2 (more accurate masks) ---
        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Preparing full-FPS tracking'})
        print(f"[SAM2] Using original FPS for SAM2 tracking (no 10fps conversion)")

        # --- 4. Generate masks at FULL FPS using WSL2 SAM2 subprocess ---
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Generating masks with SAM2 (FULL FPS via WSL2)'})

        if not points:
            raise ValueError("No points provided for mask generation.")

        # Convert points to a bounding box for the tracker
        # Accepts the following formats:
        # - list of [x, y]
        # - list of [x, y, ...] (extra values ignored)
        # - list of {x: ..., y: ...}
        # - flat list [x1, y1, x2, y2, ...] (pairs)
        # - dict with key 'bbox' as [x1, y1, x2, y2]

        def _extract_xy_array(pts):
            # dict with bbox provided directly
            if isinstance(pts, dict):
                if 'bbox' in pts and isinstance(pts['bbox'], (list, tuple)) and len(pts['bbox']) >= 4:
                    bx1, by1, bx2, by2 = pts['bbox'][:4]
                    return np.array([[bx1, by1], [bx2, by2]], dtype=float)
                if 'points' in pts:
                    pts = pts['points']
                else:
                    raise ValueError('Invalid points format: dict missing bbox/points')

            # list/tuple handling
            if isinstance(pts, (list, tuple)) and len(pts) > 0:
                first = pts[0]
                # list of dicts
                if isinstance(first, dict):
                    xy = []
                    for p in pts:
                        xv = p.get('x', p.get('X', p.get('cx')))
                        yv = p.get('y', p.get('Y', p.get('cy')))
                        if xv is None or yv is None:
                            raise ValueError('Invalid points format: dict items must contain x/y')
                        xy.append([xv, yv])
                    return np.array(xy, dtype=float)
                # list of lists/tuples/arrays
                if isinstance(first, (list, tuple)):
                    arr = np.array(pts, dtype=float)
                    # ignore extras beyond first two columns
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        return arr[:, :2]
                    # if nested but not 2D as expected
                    raise ValueError('Invalid points format: nested lists must be of shape (N, >=2)')
                # flat list of numbers
                if isinstance(first, (int, float)):
                    arr = np.array(pts, dtype=float)
                    if arr.size == 2:
                        return arr.reshape(1, 2)
                    if arr.size >= 4:
                        xs = arr[0::2]
                        ys = arr[1::2]
                        return np.stack([xs, ys], axis=1)
            raise ValueError('Invalid points format')

        xy = _extract_xy_array(points)

        # If points look normalized [0..1], scale to pixel space
        try:
            if xy.size >= 2:
                max_x = float(np.max(xy[:, 0]))
                max_y = float(np.max(xy[:, 1]))
                min_x = float(np.min(xy[:, 0]))
                min_y = float(np.min(xy[:, 1]))
                # Heuristic: values within [0,1] likely normalized
                if 0.0 <= min_x <= 1.0 and 0.0 <= max_x <= 1.0 and 0.0 <= min_y <= 1.0 and 0.0 <= max_y <= 1.0:
                    px_w = video_width or width
                    px_h = video_height or height
                    xy[:, 0] = xy[:, 0] * px_w
                    xy[:, 1] = xy[:, 1] * px_h
        except Exception:
            # Non-fatal; fallback to raw values
            pass

        x1, y1 = np.min(xy, axis=0)
        x2, y2 = np.max(xy, axis=0)

        # Ensure bbox has non-zero area; expand a bit if needed
        if int(x1) == int(x2) and int(y1) == int(y2):
            pad = 8
            x1 -= pad
            y1 -= pad
            x2 += pad
            y2 += pad

        # Clip to video bounds
        px_w = video_width or width
        px_h = video_height or height
        x1 = max(0, min(int(round(x1)), int(px_w) - 1))
        y1 = max(0, min(int(round(y1)), int(px_h) - 1))
        x2 = max(0, min(int(round(x2)), int(px_w) - 1))
        y2 = max(0, min(int(round(y2)), int(px_h) - 1))

        # Sort coords to ensure x1<=x2, y1<=y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        bbox = [x1, y1, x2, y2]
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        # Use original frame index directly (clip to bounds)
        frame_index_full = min(max(int(frame_index), 0), total_frames_original - 1)
        print(f"[SAM2] Tracking from bbox {bbox} on full-FPS frame {frame_index_full}")

        # Create output masks directory for WSL2 script (full FPS)
        masks_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks")
        os.makedirs(masks_dir, exist_ok=True)

        # Call WSL2 subprocess for SAM2 tracking (PyTorch + torch.compile)
        def _to_wsl_path(p: str) -> str:
            try:
                if len(p) >= 2 and p[1] == ':' and (p[0].isalpha()):
                    drive = p[0].lower()
                    rest = p[2:].replace('\\', '/')
                    rest = rest.lstrip('/')
                    return f"/mnt/{drive}/{rest}"
                return p.replace('\\', '/')
            except Exception:
                return p

        # Prefer WSL Celery worker (shared Redis) for mask generation using non-blocking chain
        from celery import signature, chain
        prompt_mode = os.getenv('SAM2_PROMPT_MODE', 'point').strip().lower()
        if prompt_mode == 'point':
            # Extract ALL points with labels for multi-click support
            px_w = video_width or width
            px_h = video_height or height
            points_arg = []
            labels_arg = []
            for i, (x, y) in enumerate(xy):
                px = int(np.clip(x, 0, px_w - 1))
                py = int(np.clip(y, 0, px_h - 1))
                points_arg.append((px, py))
                # Try to get label from original points array (default=1 for foreground)
                if isinstance(points, list) and i < len(points) and isinstance(points[i], dict):
                    labels_arg.append(int(points[i].get('label', 1)))
                else:
                    labels_arg.append(1)
            print(f"[SAM2] Using {len(points_arg)} POINT prompt(s): {points_arg}")
            bbox_arg = None
        else:
            print(f"[SAM2] Using BBOX prompt: {bbox_str}")
            points_arg = None
            labels_arg = None
            bbox_arg = [int(x) for x in bbox]

        if os.getenv('USE_WSL_CELERY', '1').lower() in ('1','true','yes','on'):
            print(f"[SAM2] Dispatching mask generation to WSL Celery worker (chain)…")
            # Optional: do FPS conversion on Windows side for speed (avoid requiring ffmpeg in WSL)
            track_src = video_path
            if os.getenv('SAM2_TRACK_DOWNSAMPLE', '0').lower() in ('1','true','yes','on'):
                video_downsampled_path = os.path.join(TEMP_DIR, f"{video_id}_tracking_{SAM2_TRACK_FPS}fps.mp4")
                if not os.path.exists(video_downsampled_path):
                    print(f"[SAM2] Creating {SAM2_TRACK_FPS}fps tracking video on Windows for WSL speed…")
                    convert_to_tracking_fps(video_path, video_downsampled_path)
                else:
                    print(f"[SAM2] Using cached {SAM2_TRACK_FPS}fps tracking video: {video_downsampled_path}")
                track_src = video_downsampled_path

            s1 = signature('sam2.generate_masks_fullfps', args=[track_src, masks_dir, prompt_mode, points_arg, labels_arg, bbox_arg, frame_index_full], queue='wsl_sam2')
            s2 = signature('watermark._continue_after_masks', args=[video_path, video_id, points, video_width, video_height, frame_index, api_base], queue='propainter')
            # Replace current task with the chained workflow to avoid blocking (.get) inside a task
            raise self.replace(chain(s1, s2))

        # If using WSL worker pattern (non-blocking), delegate continuation via chain
        use_wsl_celery = os.getenv('USE_WSL_CELERY', '1').lower() in ('1','true','yes','on')
        if use_wsl_celery:
            from celery import signature, chain
            prompt_mode = os.getenv('SAM2_PROMPT_MODE', 'point').strip().lower()
            if prompt_mode == 'point':
                # Extract ALL points with labels for multi-click support
                px_w = video_width or width
                px_h = video_height or height
                points_arg = []
                labels_arg = []
                for i, (x, y) in enumerate(xy):
                    px = int(np.clip(x, 0, px_w - 1))
                    py = int(np.clip(y, 0, px_h - 1))
                    points_arg.append((px, py))
                    if isinstance(points, list) and i < len(points) and isinstance(points[i], dict):
                        labels_arg.append(int(points[i].get('label', 1)))
                    else:
                        labels_arg.append(1)
                bbox_arg = None
            else:
                points_arg = None
                labels_arg = None
                bbox_arg = [int(x) for x in bbox]
            s1 = signature('sam2.generate_masks_fullfps', args=[video_path, masks_dir, prompt_mode, points_arg, labels_arg, bbox_arg, frame_index_full], queue='wsl_sam2')
            s2 = signature('watermark._continue_after_masks', args=[video_path, video_id, points, video_width, video_height, frame_index, api_base], queue='propainter')
            raise self.replace(chain(s1, s2))

        if not use_wsl_celery:
            wsl_video = _to_wsl_path(video_path)
            wsl_masks = _to_wsl_path(masks_dir)
            if prompt_mode == 'point':
                # CLI fallback only supports single point - use first point
                if len(xy) > 1:
                    print(f"[SAM2-WSL2] WARNING: CLI mode only supports 1 point, but {len(xy)} provided. Using first point only.")
                px_w = video_width or width
                px_h = video_height or height
                cx = int(np.clip(xy[0, 0], 0, px_w - 1))
                cy = int(np.clip(xy[0, 1], 0, px_h - 1))
                prompt_flag = f'--point {cx},{cy}'
            else:
                prompt_flag = f'--bbox {bbox_str}'

            wsl_cmd = (
                f'cd /mnt/d/watermarkz && '
                f'source venv_wsl2/bin/activate && '
                f'python sam2_track_wsl2.py "{wsl_video}" "{wsl_masks}" '
                f'{prompt_flag} --frame-idx {frame_index_full}'
            )
            print(f"[SAM2-WSL2] Running: wsl -e bash -c \"{wsl_cmd}\"")

            result = subprocess.run(
                ['wsl', '-e', 'bash', '-c', wsl_cmd],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"[SAM2-WSL2] STDERR: {result.stderr}")
                raise RuntimeError(f"WSL2 SAM2 tracking failed: {result.stderr}")

            print(f"[SAM2-WSL2] STDOUT: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")

            try:
                for line in result.stdout.split('\n'):
                    if '[RESULT]' in line:
                        json_str = line.split('[RESULT]')[1].strip()
                        sam2_result = json.loads(json_str)
                        print(f"[SAM2-WSL2] Result: {sam2_result}")
                        break
            except Exception as e:
                print(f"[SAM2-WSL2] Warning: Could not parse result JSON: {e}")

        # Read masks from output directory (FULL FPS)
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        if not mask_files:
            raise RuntimeError(f"No masks generated in {masks_dir}")

        masks_full = []
        for mask_file in mask_files:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks_full.append(mask)

        print(f"[SAM2-WSL2] Loaded {len(masks_full)} masks from {masks_dir}")

        # --- 5. Prepare for per-segment frame loading (DON'T load all frames upfront!) ---
        self.update_state(state='PROCESSING', meta={'progress': 15, 'status': 'Preparing segment processing'})
        total_frames = total_frames_original
        print(f"[SAM2] Video has {total_frames} frames (will load per-segment to save memory)")

        # --- 6. Use FULL-FPS masks directly (no expansion needed) ---
        self.update_state(state='PROCESSING', meta={'progress': 18, 'status': 'Validating masks'})
        if len(masks_full) != total_frames:
            print(f"[SAM2] Warning: mask count {len(masks_full)} != frame count {total_frames}; aligning by truncation")
            min_count = min(len(masks_full), total_frames)
            masks_full = masks_full[:min_count]
            total_frames = min_count
        all_masks = masks_full

        # Optionally fill short internal gaps by replicating nearest non-empty mask
        if FILL_MASK_GAPS and all_masks:
            max_gap = int(round(original_fps * MASK_GAP_MAX_SECONDS))
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            i = 0
            while i < len(presence):
                if not presence[i]:
                    # start of a gap
                    j = i
                    while j < len(presence) and not presence[j]:
                        j += 1
                    gap_len = j - i
                    if gap_len > 0 and gap_len <= max_gap:
                        # choose source: prefer previous non-empty, fallback to next
                        src_idx = i - 1 if i - 1 >= 0 and presence[i - 1] else (j if j < len(presence) else None)
                        if src_idx is not None and 0 <= src_idx < len(all_masks):
                            src = all_masks[src_idx]
                            for k in range(i, j):
                                all_masks[k] = src.copy()
                    i = j
                else:
                    i += 1

        # Optionally fill tail by replicating last non-empty mask up to a max duration
        if TAIL_MASK_FILL and all_masks:
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            if any(presence):
                last_idx = max(i for i, p in enumerate(presence) if p)
                if last_idx < (total_frames - 1):
                    max_fill = min(int(round(original_fps * TAIL_MASK_MAX_SECONDS)), (total_frames - 1) - last_idx)
                    if max_fill > 0:
                        src = all_masks[last_idx]
                        for k in range(1, max_fill + 1):
                            # replicate mask (copy) to avoid aliasing surprises
                            all_masks[last_idx + k] = src.copy()
                        print(f"[SAM2] Tail fill: replicated last mask across {max_fill} frame(s) to cover video end")

        # --- 8. Validate masks and count coverage (in memory) ---
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'Validating masks'})

        masks_with_content = sum(1 for m in all_masks if np.sum(m > 127) > 0)
        print(f"[SAM2] Mask validation: {masks_with_content}/{len(all_masks)} frames have mask content")

        if masks_with_content == 0:
            raise RuntimeError("No mask content generated - SAM2 tracking failed")

        # --- 9. Determine segment strategy ---
        if SAM2_USE_SEGMENTS:
            self.update_state(state='PROCESSING', meta={'progress': 22, 'status': 'Detecting segments'})
            if SAM2_SEGMENT_DETECTION_MODE == 'full':
                print(f"[SAM2] Detecting segments from FULL-FPS in-memory masks...")
                # Compute bbox per full-FPS mask in memory, then run detection
                detections_per_frame = []
                for m in all_masks:
                    coords = cv2.findNonZero(m)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        detections_per_frame.append((x, y, x + w, y + h))
                    else:
                        detections_per_frame.append(None)

                # Use motion-based detection for fast-moving objects (frame-to-frame comparison)
                # or average-based detection for static watermarks
                if SEGMENT_USE_MOTION_DETECTION:
                    segments = detect_segments_motion_based(
                        detections_per_frame,
                        motion_threshold=SEGMENT_MOTION_THRESHOLD,
                        min_segment_length=SEGMENT_MIN_LEN_FULL,
                        max_segment_pixels=MAX_SEGMENT_PIXELS,
                    )
                    print(f"[SAM2] Motion-based detection: {len(segments)} segments (threshold={SEGMENT_MOTION_THRESHOLD}px, max_pixels={MAX_SEGMENT_PIXELS:,})")
                else:
                    segments = detect_segments(
                        detections_per_frame,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        min_segment_length=SEGMENT_MIN_LEN_FULL,
                    )

                if len(segments) > 1 and not SEGMENT_USE_MOTION_DETECTION:
                    segments = merge_adjacent_segments(
                        segments,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        max_gap=SEGMENT_MERGE_GAP_FULL,
                    )

                # Convert inclusive end -> exclusive end and clamp to total_frames
                if segments:
                    segments = [
                        (max(0, int(s)), min(int(e) + 1, total_frames), bb) for (s, e, bb) in segments
                        if max(0, int(s)) < min(int(e) + 1, total_frames)
                    ]

                    # --- BOUNDARY-SAFE: Ensure CONTIGUOUS coverage (no gaps!) ---
                    # Empty-mask frames still processed but with zero mask (no inpainting)
                    fixed_segments = []
                    for i, (s, e, bb) in enumerate(segments):
                        if i == 0 and s > 0:
                            s = 0  # First segment must start at 0
                        if i > 0:
                            prev_end = fixed_segments[-1][1]
                            if s > prev_end:
                                s = prev_end  # No gaps between segments!
                        if i == len(segments) - 1:
                            e = total_frames  # Last segment must reach the end
                        fixed_segments.append((s, e, bb))
                    segments = fixed_segments

                    # Replace each segment's averaged bbox with union bbox for that segment's frames
                    # This ensures fast-moving objects are fully covered within each segment
                    # Also splits segments if union bbox exceeds MAX_SEGMENT_PIXELS
                    if detections_per_frame:
                        updated_segments = []
                        for s, e, bb in segments:
                            # Compute union of all bboxes in this segment's frame range
                            segment_bboxes = [detections_per_frame[f] for f in range(s, e)
                                             if f < len(detections_per_frame) and detections_per_frame[f] is not None]
                            if segment_bboxes:
                                seg_min_x = min(b[0] for b in segment_bboxes)
                                seg_min_y = min(b[1] for b in segment_bboxes)
                                seg_max_x = max(b[2] for b in segment_bboxes)
                                seg_max_y = max(b[3] for b in segment_bboxes)
                                union_bbox = (seg_min_x, seg_min_y, seg_max_x, seg_max_y)

                                # Check if CROP (post-padding) exceeds pixel limit
                                # Use calculate_crop_region to get EXACT crop size (includes min_size, clamping)
                                _, _, crop_w, crop_h = calculate_crop_region(
                                    union_bbox, width, height,
                                    padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                                )
                                crop_pixels = crop_w * crop_h
                                if MAX_CROP_PIXELS > 0 and crop_pixels > MAX_CROP_PIXELS:
                                    # Split this segment into smaller pieces
                                    print(f"[SAM2] Segment {s}-{e}: crop {crop_pixels:,} px ({crop_w}x{crop_h}) > {MAX_CROP_PIXELS:,} limit, splitting...")
                                    sub_segments = split_segment_by_pixels(s, e, detections_per_frame, MAX_CROP_PIXELS, SEGMENT_CROP_PAD_RATIO, width, height)
                                    for sub_s, sub_e, sub_bb in sub_segments:
                                        _, _, sub_w, sub_h = calculate_crop_region(
                                            sub_bb, width, height,
                                            padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                                        )
                                        sub_crop_px = sub_w * sub_h
                                        print(f"[SAM2]   -> Sub-segment {sub_s}-{sub_e} ({sub_e-sub_s}f): crop {sub_crop_px:,} px ({sub_w}x{sub_h})")
                                    updated_segments.extend(sub_segments)
                                else:
                                    updated_segments.append((s, e, union_bbox))
                                    if union_bbox != bb:
                                        print(f"[SAM2] Segment {len(updated_segments)}: union bbox {union_bbox} (was avg {bb})")
                            else:
                                updated_segments.append((s, e, bb))
                        segments = updated_segments

                # Split segments that exceed MAX_SEGMENT_FRAMES to prevent OOM
                if MAX_SEGMENT_FRAMES > 0:
                    split_segments = []
                    for s, e, bb in segments:
                        seg_len = e - s
                        if seg_len > MAX_SEGMENT_FRAMES:
                            # Split into chunks of MAX_SEGMENT_FRAMES
                            num_chunks = (seg_len + MAX_SEGMENT_FRAMES - 1) // MAX_SEGMENT_FRAMES
                            chunk_size = seg_len // num_chunks
                            for i in range(num_chunks):
                                chunk_start = s + i * chunk_size
                                chunk_end = s + (i + 1) * chunk_size if i < num_chunks - 1 else e
                                split_segments.append((chunk_start, chunk_end, bb))
                            print(f"[SAM2] Split {seg_len}f segment into {num_chunks} chunks of ~{chunk_size}f")
                        else:
                            split_segments.append((s, e, bb))
                    segments = split_segments

                # --- POST-PROCESS: Remove empty mask frames and split segments ---
                MIN_FRAMES_SPLIT = 5  # Min frames for segments broken by empty masks

                def filter_empty_frames_from_segments(segs, masks, det_per_frame):
                    """Remove empty mask frames, split if empty in middle."""
                    filtered = []
                    for s, e, bb in segs:
                        # Find runs of non-empty frames
                        runs = []
                        run_start = None
                        for f in range(s, e):
                            has_content = f < len(masks) and np.sum(masks[f] > 127) > 0
                            if has_content:
                                if run_start is None:
                                    run_start = f
                            else:
                                if run_start is not None:
                                    runs.append((run_start, f))
                                    run_start = None
                        if run_start is not None:
                            runs.append((run_start, e))

                        # Create segments from runs
                        is_split = len(runs) > 1
                        min_len = MIN_FRAMES_SPLIT if is_split else SEGMENT_MIN_LEN_FULL
                        for run_s, run_e in runs:
                            if run_e - run_s >= min_len:
                                # Compute bbox for this run
                                run_bboxes = [det_per_frame[f] for f in range(run_s, run_e)
                                              if f < len(det_per_frame) and det_per_frame[f]]
                                if run_bboxes:
                                    run_bb = (min(b[0] for b in run_bboxes), min(b[1] for b in run_bboxes),
                                              max(b[2] for b in run_bboxes), max(b[3] for b in run_bboxes))
                                else:
                                    run_bb = bb
                                filtered.append((run_s, run_e, run_bb))
                                if is_split:
                                    print(f"[SAM2] Empty-mask split: frames {run_s}-{run_e-1} ({run_e-run_s}f)")
                    return filtered

                orig_count = len(segments)
                segments = filter_empty_frames_from_segments(segments, all_masks, detections_per_frame)
                if len(segments) != orig_count:
                    print(f"[SAM2] Filtered empty masks: {orig_count} -> {len(segments)} segments")

                print(f"[SAM2] Detected {len(segments)} segments (FULL-FPS, boundary-safe)")
                for idx, (start, end, bbox) in enumerate(segments):
                    print(f"[SAM2]   Segment {idx+1}: frames {start}-{end-1} ({end-start}f) bbox={bbox}")
            else:
                print(f"[SAM2] Detecting segments from 10fps masks...")

                segments_10fps = detect_segments_from_masks(
                    masks_10fps_dir,
                    position_tolerance=SEGMENT_POS_TOLERANCE,
                    min_segment_length=SEGMENT_MIN_LEN_10FPS,
                    max_segments=80,
                )

                if len(segments_10fps) > 1:
                    segments_10fps = merge_adjacent_segments(
                        segments_10fps,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        max_gap=SEGMENT_MERGE_GAP_10FPS,
                    )

                # Scale segments from 10fps to full FPS
                fps_ratio = original_fps / 10.0
                segments = []
                for start_10fps, end_10fps, bbox in segments_10fps:
                    start_full = int(start_10fps * fps_ratio)
                    end_full = min(int((end_10fps + 1) * fps_ratio), total_frames)  # +1 because end is exclusive
                    segments.append((start_full, end_full, bbox))

                print(f"[SAM2] Detected {len(segments)} segments (scaled to full FPS)")
                for idx, (start, end, bbox) in enumerate(segments):
                    print(f"[SAM2]   Segment {idx+1}: frames {start}-{end-1} ({end-start}f) bbox={bbox}")
        else:
            # Strict monolithic mode (commit logic): process entire video in a single pass
            segments = [(0, total_frames, [0, 0, width, height])]
            print(f"[SAM2] Strict full-video mode: 1 segment covering all frames")

        # --- 10. Calculate global crop region from all masks ---
        print(f"[SAM2] Calculating global crop region...")
        min_x, min_y = width, height
        max_x, max_y = 0, 0

        for mask in all_masks:
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

        # With validation at line 644-648, we should always have a valid bbox
        if max_x <= min_x or max_y <= min_y:
            raise RuntimeError("No valid bounding box found despite having mask content - unexpected state")
        global_bbox = [min_x, min_y, max_x, max_y]

        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(global_bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"[SAM2] Global crop: {crop_x},{crop_y} {crop_w}x{crop_h}")

        # Define output_dir for cleanup tracking (not created - return_frames=True skips disk I/O)
        output_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_output")

        # --- 11. Process segments with ProPainter ---
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': f'Processing {len(segments)} segments'})

        faster_propainter_pipeline = get_propainter_pipeline()
        import torch
        use_fp16 = torch.cuda.is_available()

        print(f"[SAM2] ProPainter config:")
        print(f"   - Segments: {len(segments)}")
        print(f"   - Workers: {SEGMENT_WORKERS}")
        print(f"   - FP16: {use_fp16}")
        print(f"   - Optical Flow: NeuFlow TRT (USE_NEUFLOW={os.getenv('USE_NEUFLOW', '0')})")

        # Crop masks to global crop region (masks are small - 1 channel)
        # Frames will be loaded per-segment to avoid OOM on large videos
        all_masks_cropped = [m[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for m in all_masks]
        global_crop = (crop_x, crop_y, crop_w, crop_h)
        print(f"[SAM2] Masks cropped, frames will be loaded per-segment")

        # Track segment metadata
        segment_results = {}  # seg_idx -> (success, crop_info, time_info)
        processed_frames = set()  # Track which frames have been processed

        # Store final composited frames (raw numpy - no compression overhead!)
        # Only filled as segments are processed
        vram_frames = {}  # {frame_idx: numpy_bgr_frame}

        # NO PRE-LOADING! Load frames per-segment to minimize VRAM (like hasan branch)
        print(f"[SAM2] Per-segment loading mode (low VRAM) - {total_frames} frames total")

        def process_segment(seg_idx, start_f, end_f, seg_bbox):
            """Process a single segment with ProPainter - loads frames per-segment (low VRAM)
            Uses dedicated CUDA stream for parallel execution (like hasan branch)"""
            proc_start = start_f
            proc_end = end_f

            # Create dedicated CUDA stream for this segment (enables TRUE parallel GPU execution)
            cuda_stream = None
            if torch.cuda.is_available():
                cuda_stream = torch.cuda.Stream()

            try:
                # Run entire segment processing on dedicated CUDA stream
                with torch.cuda.stream(cuda_stream) if cuda_stream else nullcontext():
                    duration = end_f - start_f
                    log_end = end_f - 1 if end_f > start_f else end_f
                    # Calculate segment-specific crop directly from seg_bbox (in original frame coords)
                    seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                        list(seg_bbox),  # Use segment bbox directly in frame coordinates
                        width, height, padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                    )

                    # Hasan branch used FIXED values for speed (not resolution-based)
                    subvideo_length = 120  # Hasan: 120 fixed (larger batch = faster)
                    neighbor_length = 10   # Hasan: 10 (more temporal context)
                    ref_stride = 10        # Hasan: 10 (not 15)

                    # Temporal padding for context
                    pad_left = min(SEGMENT_TEMPORAL_PAD, start_f)
                    pad_right = min(SEGMENT_TEMPORAL_PAD, total_frames - end_f)
                    proc_start = start_f - pad_left
                    proc_end = end_f + pad_right

                    print(f"[{mode.upper()}] Segment {seg_idx+1}: frames {start_f}-{log_end} ({duration}f), crop={seg_crop_w}x{seg_crop_h}, pad=+/-{SEGMENT_TEMPORAL_PAD} => proc {proc_start}-{proc_end}")

                    # LOAD FRAMES DIRECTLY FROM VIDEO - CROPPED IMMEDIATELY (like hasan branch!)
                    # This minimizes VRAM: only segment+padding frames, already cropped
                    print(f"[{mode.upper()}] Loading {proc_end - proc_start} frames from video (cropped to {seg_crop_w}x{seg_crop_h})")
                    seg_frames = []
                    seg_masks = []
                    full_frames_for_paste = []  # Keep full frames ONLY for segment range (paste-back)

                    cap_seg = cv2.VideoCapture(video_path)
                    cap_seg.set(cv2.CAP_PROP_POS_FRAMES, proc_start)

                    for frame_idx in range(proc_start, proc_end):
                        ret, full_frame = cap_seg.read()
                        if not ret:
                            break
                        # Crop frame IMMEDIATELY (before storing) - saves VRAM!
                        cropped_frame = full_frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                        seg_frames.append(np.ascontiguousarray(cropped_frame))

                        # Keep full frame ONLY for core segment frames (for paste-back)
                        if start_f <= frame_idx < end_f:
                            full_frames_for_paste.append((frame_idx, full_frame))

                        # Crop mask
                        if frame_idx < len(all_masks):
                            seg_mask = all_masks[frame_idx][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                            seg_masks.append(np.ascontiguousarray(seg_mask))

                    cap_seg.release()

                    if not seg_frames or not seg_masks:
                        return seg_idx, None, None, (proc_start, proc_end)

                    # Run ProPainter on segment - ZERO DISK I/O with return_frames=True!
                    output_frames = faster_propainter_pipeline(
                        video='dummy',
                        mask='dummy',
                        output=output_dir,  # Not used when return_frames=True
                        resize_ratio=1.0,
                        mask_dilation=SAM2_MASK_DILATION,
                        ref_stride=ref_stride,  # Hasan: 10
                        neighbor_length=neighbor_length,
                        subvideo_length=subvideo_length,
                        raft_iter=10,
                        mode="video_inpainting",
                        save_fps=int(original_fps),
                        save_frames=False,
                        fp16=use_fp16,
                        use_cached_models=True,
                        frames_array=seg_frames,
                        masks_array=seg_masks,
                        return_frames=True  # ZERO DISK I/O!
                    )

                    if not output_frames:
                        print(f"[ERROR] Segment {seg_idx}: ProPainter returned no frames")
                        return seg_idx, False, None, (proc_start, proc_end)

                    # Composite ProPainter output onto original frames (paste-back)
                    # Only save CORE frames (start_f to end_f), skip padding frames
                    core_count = end_f - start_f
                    print(f"[{mode.upper()}] Segment {seg_idx+1}: compositing {core_count} core frames...")
                    frames_stored = 0

                    # Use the full frames we kept for paste-back
                    paste_idx = 0
                    for i in range(len(output_frames)):
                        frame_idx = proc_start + i
                        # Skip padding frames - only save core frames [start_f, end_f)
                        if frame_idx < start_f or frame_idx >= end_f:
                            continue

                        # Get original full frame from our saved list
                        if paste_idx < len(full_frames_for_paste):
                            saved_frame_idx, orig_frame = full_frames_for_paste[paste_idx]
                            paste_idx += 1

                            # Composite segment output onto original frame
                            orig_frame[seg_crop_y:seg_crop_y + seg_crop_h,
                                      seg_crop_x:seg_crop_x + seg_crop_w] = output_frames[i]
                            # Store as raw numpy (no compression - simpler, faster!)
                            vram_frames[frame_idx] = orig_frame
                            processed_frames.add(frame_idx)
                            frames_stored += 1

                    # Free memory immediately!
                    del output_frames
                    del full_frames_for_paste
                    del seg_frames
                    del seg_masks
                    print(f"[{mode.upper()}] Segment {seg_idx+1}: stored {frames_stored} core frames")

                    # Synchronize CUDA stream before returning
                    if cuda_stream:
                        cuda_stream.synchronize()

                    # Return success flag and metadata (NOT frames!)
                    return seg_idx, True, (seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h), (proc_start, proc_end)

            except Exception as e:
                print(f"[ERROR] Segment {seg_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                return seg_idx, False, None, (proc_start, proc_end)

        # Process segments (sequentially for GPU, parallel prep could be added)
        if len(segments) <= 1:
            # Single segment - load frames directly from video (cropped)
            print(f"[{mode.upper()}] Processing as single segment...")
            print(f"[{mode.upper()}] Loading {total_frames} frames from video (cropped to {crop_w}x{crop_h})")

            all_frames_contiguous = []
            full_frames_for_final = []  # Keep full frames for paste-back

            cap_single = cv2.VideoCapture(video_path)
            for frame_idx in range(total_frames):
                ret, full_frame = cap_single.read()
                if not ret:
                    break
                # Crop immediately
                cropped = full_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                all_frames_contiguous.append(np.ascontiguousarray(cropped))
                full_frames_for_final.append(full_frame)
            cap_single.release()

            all_masks_contiguous = [np.ascontiguousarray(m) for m in all_masks_cropped]

            # Hasan branch used FIXED values for speed
            subvideo_length = 120  # Hasan: 120 fixed
            neighbor_length = 10   # Hasan: 10
            ref_stride = 10        # Hasan: 10

            # Run ProPainter
            final_cropped_frames = faster_propainter_pipeline(
                video=video_path,
                mask='dummy_mask',
                output=output_dir,
                resize_ratio=1.0,
                mask_dilation=SAM2_MASK_DILATION,
                ref_stride=ref_stride,
                neighbor_length=neighbor_length,
                subvideo_length=subvideo_length,
                raft_iter=10,
                mode="video_inpainting",
                save_fps=int(original_fps),
                save_frames=False,
                fp16=use_fp16,
                use_cached_models=True,
                frames_array=all_frames_contiguous,
                masks_array=all_masks_contiguous,
                return_frames=True
            )

            if not final_cropped_frames:
                raise RuntimeError("ProPainter returned no frames")

            # Composite back onto full frames and store
            for i, cropped_result in enumerate(final_cropped_frames):
                if i < len(full_frames_for_final):
                    full_frames_for_final[i][crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cropped_result
                    vram_frames[i] = full_frames_for_final[i]
                    processed_frames.add(i)

            del all_frames_contiguous
            del final_cropped_frames

        else:
            # Multiple segments - process each (optionally in parallel) and merge
            if SAM2_PARALLEL_SEGMENTS and SEGMENT_WORKERS > 1:
                print(f"[SAM2] Processing {len(segments)} segments in PARALLEL (workers={min(SEGMENT_WORKERS, len(segments))})...")
                self.update_state(state='PROCESSING', meta={'progress': 30, 'status': f'Processing {len(segments)} segments (parallel)'})

                max_workers = min(SEGMENT_WORKERS, len(segments))
                futures = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                        futures[executor.submit(process_segment, seg_idx, start_f, end_f, seg_bbox)] = seg_idx

                    completed = 0
                    for future in as_completed(futures):
                        seg_idx_done = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            print(f"[ERROR] Segment {seg_idx_done} raised: {e}")
                            result = (seg_idx_done, None, None, (0, 0))
                        segment_results[seg_idx_done] = result
                        completed += 1
                        progress = 30 + int((completed / len(segments)) * 40)
                        self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'Processed {completed}/{len(segments)} segments'})
            else:
                print(f"[SAM2] Processing {len(segments)} segments (sequential)...")
                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                    progress = 30 + int((seg_idx / len(segments)) * 40)
                    self.update_state(state='PROCESSING', meta={
                        'progress': progress,
                        'status': f'Processing segment {seg_idx+1}/{len(segments)}'
                    })

                    clear_gpu_memory()
                    result = process_segment(seg_idx, start_f, end_f, seg_bbox)
                    segment_results[seg_idx] = result

        # --- 12. Encode from stored frames → FFmpeg pipe → NVENC ---
        # (Common path for both single-segment and multi-segment)
        print(f"[{mode.upper()}] Segment processing complete - {len(vram_frames)} frames stored")
        print(f"[{mode.upper()}] Encoding via FFmpeg pipe (NVENC)...")
        self.update_state(state='PROCESSING', meta={'progress': 75, 'status': 'Encoding (NVENC)'})
        import time as _time
        _encode_start = _time.time()

        output_path = os.path.join(RESULT_DIR, f"{video_id}_sam2_removed.mp4")

        # Get frame dimensions (use original video dimensions)
        frame_height, frame_width = height, width

        # FFmpeg command: pipe rawvideo → NVENC → MP4 with audio from original
        ffmpeg_cmd = [
            str(FFMPEG_EXE), '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{frame_width}x{frame_height}',
            '-r', str(int(original_fps)),
            '-i', 'pipe:0',  # Video from stdin
            '-i', video_path,  # Audio from original
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'h264_nvenc',  # GPU encoding!
            '-preset', 'p1',  # Fastest NVENC preset
            '-b:v', '8M',
            '-bufsize', '16M',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]

        # Start FFmpeg process
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # For frames not in vram_frames, read from original video
        cap_encode = cv2.VideoCapture(video_path)

        # Stream ALL frames directly to FFmpeg (processed + unprocessed)
        for frame_idx in range(total_frames):
            # Progress logging every 100 frames
            if frame_idx % 100 == 0:
                print(f"[{mode.upper()}] Encoding frame {frame_idx}/{total_frames}...")

            if frame_idx in vram_frames:
                # Processed frame from segment - already composited
                frame_bgr = vram_frames[frame_idx]
            else:
                # Unprocessed frame - read from original video
                ret, frame_bgr = cap_encode.read()
                if not ret:
                    print(f"[WARNING] Frame {frame_idx} read failed, skipping")
                    continue

            proc.stdin.write(frame_bgr.tobytes())

        cap_encode.release()

        proc.stdin.close()
        proc.wait(timeout=300)

        if proc.returncode != 0:
            print(f"[ERROR] FFmpeg NVENC encoding failed (code {proc.returncode})")
            raise RuntimeError(f"Video encoding failed (code {proc.returncode})")

        _encode_time = _time.time() - _encode_start
        print(f"[SAM2] NVENC encode complete: {total_frames} frames in {_encode_time:.2f}s ({total_frames/_encode_time:.1f} fps)")

        # Free VRAM
        import torch as _torch
        del vram_frames
        _torch.cuda.empty_cache()

        print(f"[SAM2] Final video: {output_path}")

        # --- 15b. Upload to B2 + Cloudflare CDN (inlined - no heavy imports!) ---
        cdn_url = None
        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo
            import time as _upload_time

            B2_KEY_ID = os.getenv('B2_KEY_ID', '00539db5c1104b50000000002')
            B2_APP_KEY = os.getenv('B2_APP_KEY', 'K005HJKUP7ahSNJ1wgQHDDJ+uEATiU4')
            B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
            B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

            if os.getenv('B2_UPLOAD_ENABLED', '1') == '1':
                timestamp = int(_upload_time.time())
                remote_path = f"results/{timestamp}_{os.path.basename(output_path)}"

                print(f"[B2] Uploading to {B2_BUCKET}/{remote_path}...")
                _b2_start = _upload_time.time()
                info = InMemoryAccountInfo()
                b2_api = B2Api(info)
                b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
                bucket = b2_api.get_bucket_by_name(B2_BUCKET)
                bucket.upload_local_file(local_file=output_path, file_name=remote_path)
                cdn_url = f"{B2_CDN_URL}/{remote_path}"
                _b2_time = _upload_time.time() - _b2_start
                print(f"[B2] Upload complete in {_b2_time:.1f}s - CDN URL: {cdn_url}")
        except ImportError:
            print(f"[B2] b2sdk not installed - skipping upload")
        except Exception as e:
            print(f"[B2] Upload failed: {e}")

        # --- 16. Cleanup temp files ---
        print(f"[SAM2] Cleaning up...")
        for temp_path in [output_dir, masks_dir]:
            if isinstance(temp_path, str) and os.path.exists(temp_path):
                if os.path.isdir(temp_path):
                    shutil.rmtree(temp_path)
                else:
                    os.remove(temp_path)

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})

        print(f"[SAM2] Complete! Output: {output_path}")

        return {
            'status': 'success',
            'video_id': video_id,
            'video_path': video_path,
            'output_path': cdn_url or output_path,
            'total_frames': total_frames,
            'masks_generated': len(all_masks),
            'masks_expanded': len(all_masks),
            'segments_processed': len(segments),
            'width': width,
            'height': height,
            'fps': original_fps,
            'pipeline': 'full_fps_segments',
            'message': f'SAM2 pipeline complete! Processed {len(segments)} segment(s)'
        }

    except Exception as e:
        print(f"[ERROR] SAM2 task failed: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup temp files on failure too
        try:
            for temp_path in [output_dir, masks_dir]:
                if isinstance(temp_path, str) and os.path.exists(temp_path):
                    if os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                        print(f"[CLEANUP] Removed {temp_path}")
        except Exception as cleanup_err:
            print(f"[CLEANUP] Warning: {cleanup_err}")
        # Do not call update_state with FAILURE meta (Celery will store proper exception info)
        raise


#============================================================================
# YOLO WINDOWS TASK (runs entirely on Windows - no WSL!)
#============================================================================

@celery.task(bind=True, name='yolo.process_video_windows')
def process_video_yolo_windows(self, video_path, video_id=None, api_base=None):
    """
    YOLO watermark removal - runs entirely on Windows with batch TensorRT.

    This is a port of the main branch's YOLO logic:
    - Batch YOLO detection (64 frames at once) - ~1-2ms per frame
    - In-memory frame/mask storage (FRAME_CACHE)
    - Position-based segment detection
    - Parallel ProPainter inpainting

    Much faster than WSL chain (no network round-trips, no B2 upload/download).
    """
    import shutil
    from pathlib import Path

    try:
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Starting YOLO detection'})

        # Load detector (TensorRT batch engine)
        yolo_detector = get_detector()

        # Verify video exists
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video not found: {video_path}")

        print(f"[YOLO] Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        base_name = Path(video_path).stem
        video_id = video_id or str(uuid.uuid4())[:8]

        print(f"[YOLO] Video: {width}x{height} @ {fps} fps ({total_frames} frames)")

        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': f'Loading {total_frames} frames'})

        # Load all frames to memory for batch processing
        print(f"[YOLO] Loading {total_frames} frames to memory...")
        decode_start = time.time()

        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        frames_processed = len(all_frames)
        decode_time = time.time() - decode_start
        print(f"[YOLO] Decoded {frames_processed} frames: {decode_time:.2f}s ({decode_time/frames_processed*1000:.2f}ms/frame)")

        if frames_processed == 0:
            raise RuntimeError("No frames decoded from video")

        # BATCH DETECTION (EXTREME SPEED - 1-2ms per frame!)
        self.update_state(state='PROCESSING', meta={'progress': 15, 'status': 'Batch YOLO detection (1-2ms/frame)'})

        print(f"[YOLO] Running batch detection on {frames_processed} frames...")
        batch_start = time.time()
        all_detections = yolo_detector.detect_batch(all_frames, confidence_threshold=0.15, padding=0, batch_size=64)
        batch_duration = time.time() - batch_start
        ms_per_frame = (batch_duration / max(frames_processed, 1)) * 1000
        print(f"[YOLO] Batch detection: {batch_duration:.2f}s ({ms_per_frame:.2f}ms/frame)")

        # Create masks and track bboxes
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': 'Creating masks'})

        zero_mask = np.zeros((height, width), dtype=np.uint8)
        bboxes_per_frame = []
        frames_with_watermark = 0
        last_valid_bbox = None
        all_masks = []

        for i, detections in enumerate(all_detections):
            if detections:
                frames_with_watermark += 1
                last_valid_bbox = detections[0]['bbox']
                bboxes_per_frame.append(last_valid_bbox)
                # Create mask from detection
                mask = yolo_detector.create_mask(all_frames[i], detections)
                all_masks.append(mask)
            elif last_valid_bbox:
                bboxes_per_frame.append(last_valid_bbox)
                # Use previous mask (carry forward)
                all_masks.append(all_masks[-1] if all_masks else zero_mask)
            else:
                bboxes_per_frame.append(None)
                all_masks.append(zero_mask)

        print(f"[YOLO] Created {len(all_masks)} masks, {frames_with_watermark} frames with watermarks")

        if frames_with_watermark == 0:
            raise RuntimeError("No watermarks detected in video")

        # Store in global FRAME_CACHE for instant access by segment workers
        cache_key = f"video_data:{base_name}"
        with FRAME_CACHE_LOCK:
            FRAME_CACHE[cache_key] = {
                'frames': all_frames,
                'masks': all_masks,
                'bboxes': bboxes_per_frame,
                'timestamp': time.time(),
                'video_id': video_id,
                'base_name': base_name
            }
        print(f"[YOLO] Stored {frames_processed} frames/masks in FRAME_CACHE")

        # Detect segments (position-based, like main branch)
        self.update_state(state='PROCESSING', meta={'progress': 40, 'status': 'Detecting segments'})

        segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
        if segments:
            segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)

        print(f"[YOLO] Detected {len(segments)} segments")

        if not segments:
            # Fallback: single segment covering all frames
            segments = [(0, frames_processed)]
            print(f"[YOLO] No segments detected, using full video as single segment")

        # Create output directories
        output_dir = os.path.join(RESULT_DIR, f"{base_name}_{video_id}")
        masks_dir = os.path.join(TEMP_DIR, f"{video_id}_yolo_masks")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        # Save masks to disk for ProPainter (it needs file paths)
        print(f"[YOLO] Saving masks to {masks_dir}...")
        for i, mask in enumerate(all_masks):
            cv2.imwrite(os.path.join(masks_dir, f"{i:04d}.png"), mask)

        # Now call the continuation task (same as SAM2 flow)
        # Build sam2_result dict to match expected format
        sam2_result = {
            'masks_dir': masks_dir,
            'mode': 'yolo',
            'total_frames': frames_processed,
            'frames_with_watermark': frames_with_watermark,
            'segments': [list(s) for s in segments],
            'bboxes': bboxes_per_frame,
            'width': width,
            'height': height,
            'fps': fps
        }

        self.update_state(state='PROCESSING', meta={'progress': 50, 'status': 'Starting ProPainter inpainting'})

        # Call continuation task synchronously using .apply() (same worker, uses FRAME_CACHE)
        # .apply() executes the task synchronously in the current process
        task_result = _continue_after_masks.apply(
            args=[sam2_result, video_path, video_id, None, width, height, 0, api_base]
        )
        result = task_result.get()  # Wait for result

        # Cleanup FRAME_CACHE after processing
        with FRAME_CACHE_LOCK:
            if cache_key in FRAME_CACHE:
                del FRAME_CACHE[cache_key]
                print(f"[YOLO] Cleaned up FRAME_CACHE: {cache_key}")

        return result

    except Exception as e:
        print(f"[ERROR] YOLO task failed: {e}")
        import traceback
        traceback.print_exc()
        raise


#============================================================================
# FLASK ROUTES
#============================================================================

@app.route('/api/process_sam2', methods=['POST'])
def process_sam2():
    """
    API endpoint to process video with SAM2 masks

    POST data:
    {
        "video_path": "D:\\watermarkz\\videostotrain\\stock2.mp4",
        "masks_folder": "D:\\watermarkz\\temp_sam2_masks",
        "video_id": "stock2"  # Optional
    }
    """
    data = request.get_json()
    video_path = data.get('video_path')
    masks_folder = data.get('masks_folder')
    video_id = data.get('video_id')

    if not video_path or not masks_folder:
        return jsonify({'error': 'Missing video_path or masks_folder'}), 400

    # Queue Celery task
    result = celery.send_task(
        'watermark.process_sam2_interactive',
        args=[video_path, masks_folder, video_id]
    )

    return jsonify({
        'task_id': result.id,
        'status': 'queued',
        'message': 'SAM2 processing task queued'
    })


@app.route('/api/process_yolo', methods=['POST'])
def process_yolo():
    """
    API endpoint to process video with YOLO detection (automatic, no user clicks).

    POST data:
    {
        "video_path": "D:\\watermarkz\\uploads\\video.mp4",
        "video_id": "abc123"  # Optional
    }

    NEW: Runs entirely on Windows with batch TensorRT (no WSL chain!)
    - Batch YOLO detection: 64 frames at once, ~1-2ms per frame
    - In-memory frame/mask storage (FRAME_CACHE)
    - Much faster than WSL chain (no network round-trips)

    YOLO mode uses 4 parallel segment workers (vs 2 for SAM2).
    """
    data = request.get_json()
    video_path = data.get('video_path')
    video_id = data.get('video_id') or str(uuid.uuid4())[:8]

    if not video_path:
        return jsonify({'error': 'Missing video_path'}), 400

    if not os.path.exists(video_path):
        return jsonify({'error': f'Video not found: {video_path}'}), 404

    # Get API base URL for result upload
    api_base = os.getenv('TUNNEL_URL', os.getenv('API_BASE_URL', 'http://localhost:5001'))

    # NEW: Run entirely on Windows (no WSL chain!)
    result = process_video_yolo_windows.apply_async(
        args=[video_path, video_id, api_base],
        queue='propainter'  # Uses same queue as ProPainter (Windows GPU worker)
    )

    print(f"[YOLO] Queued Windows YOLO task: {result.id}")
    print(f"  - Video: {video_path}")
    print(f"  - Mode: Windows TensorRT (batch 64, ~1-2ms/frame)")

    return jsonify({
        'task_id': result.id,
        'status': 'queued',
        'message': 'YOLO processing task queued (Windows TensorRT)',
        'mode': 'yolo'
    })


@app.route('/api/process_yolo_parallel', methods=['POST'])
def process_yolo_parallel():
    """
    API endpoint to process video with YOLO + Celery chord pattern (FAST parallel mode).

    POST data:
    {
        "video_path": "D:\\watermarkz\\uploads\\video.mp4",
        "video_id": "abc123"  # Optional
    }

    This uses the NEW parallel processing pattern:
    - Celery chord dispatches segments to multiple workers
    - Background encoder encodes segments as they complete
    - 4 parallel workers = ~13s for 300 frames (vs 38s sequential)
    """
    data = request.get_json()
    video_path = data.get('video_path')
    video_id = data.get('video_id') or str(uuid.uuid4())[:8]

    if not video_path:
        return jsonify({'error': 'Missing video_path'}), 400

    if not os.path.exists(video_path):
        return jsonify({'error': f'Video not found: {video_path}'}), 404

    # Get API base URL for result upload
    api_base = os.getenv('TUNNEL_URL', os.getenv('API_BASE_URL', 'http://localhost:5001'))

    # Use new parallel YOLO task with chord pattern
    result = yolo_prepare_video_task.apply_async(
        args=[video_path],
        kwargs={'api_base': api_base, 'video_id': video_id},
        queue='yolo'  # Uses dedicated yolo queue with threads pool
    )

    print(f"[YOLO PARALLEL] Queued parallel YOLO task: {result.id}")
    print(f"  - Video: {video_path}")
    print(f"  - Mode: Celery chord + background encoder (4 parallel segments)")

    return jsonify({
        'task_id': result.id,
        'status': 'queued',
        'message': 'YOLO parallel processing task queued (chord + background encoder)',
        'mode': 'yolo_parallel',
        'video_id': video_id,
    })


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a Celery task"""
    task = celery.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('progress', 0),
            'status': task.info.get('status', ''),
            'result': task.info if task.state == 'SUCCESS' else None
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }

    return jsonify(response)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'server': 'SAM2 Production'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


#============================================================================
# CONTINUATION TASK (runs after WSL mask generation)
#============================================================================

@celery.task(bind=True, name='watermark._continue_after_masks')
def _continue_after_masks(self, sam2_result, video_path, video_id=None, points=None, video_width=None, video_height=None, frame_index=0, api_base=None):
    try:
        import cv2, glob, numpy as np, os, json, zipfile, requests
        from watermark import expand_masks_10fps  # not used but keep import parity
        from urllib.parse import urljoin

        if not sam2_result or not isinstance(sam2_result, dict):
            raise RuntimeError(f"Invalid sam2_result: {sam2_result}")

        # Determine mode and adjust SEGMENT_WORKERS
        # YOLO mode: 4 workers (faster, simpler masks)
        # SAM2 mode: 2 workers (more complex tracking)
        mode = sam2_result.get('mode', 'sam2')
        if mode == 'yolo':
            os.environ['SEGMENT_WORKERS'] = '4'
            print(f"[MODE] YOLO mode detected - using 4 segment workers")
        else:
            os.environ['SEGMENT_WORKERS'] = '2'
            print(f"[MODE] SAM2 mode detected - using 2 segment workers")

        # Check for B2 URL (new flow) or local path (legacy)
        masks_url = sam2_result.get('masks_url')
        masks_dir = sam2_result.get('masks_dir')

        if masks_url and not masks_dir:
            # Download masks from B2 CDN
            print(f"[B2] Downloading masks from {masks_url}...")
            self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Downloading masks from B2'})

            zip_path = os.path.join(TEMP_DIR, f"{video_id}_masks.zip")
            r = requests.get(masks_url, timeout=120)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                f.write(r.content)

            # Extract masks
            masks_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks")
            os.makedirs(masks_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(masks_dir)
            os.remove(zip_path)

            mask_count = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])
            print(f"[B2] Extracted {mask_count} masks to {masks_dir}")
        elif not masks_dir:
            masks_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks")

        if not os.path.isdir(masks_dir):
            raise RuntimeError(f"Masks directory missing: {masks_dir}")


        # If video is a Railway path (/data/...), download it from api_base
        if video_path.startswith('/data/') and api_base:
            filename = os.path.basename(video_path)
            download_url = f"{api_base}/uploads/{filename}"
            local_video = os.path.join(TEMP_DIR, filename)
            print(f"[RECEIVER] Downloading video from {download_url}...")
            self.update_state(state='PROCESSING', meta={'progress': 8, 'status': 'Downloading video'})
            r = requests.get(download_url, timeout=300)
            r.raise_for_status()
            with open(local_video, 'wb') as f:
                f.write(r.content)
            video_path = local_video
            print(f"[RECEIVER] Downloaded {len(r.content) / 1024 / 1024:.1f} MB to {local_video}")

        # Video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Read masks
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        if not mask_files:
            raise RuntimeError(f"No masks generated in {masks_dir}")
        masks_full = [cv2.imread(mf, cv2.IMREAD_GRAYSCALE) for mf in mask_files]
        masks_full = [m for m in masks_full if m is not None]

        # DON'T load all frames upfront - use per-segment loading
        total_frames = total_frames_original
        print(f"[SAM2] Video has {total_frames} frames (will load per-segment)")

        # Align and expand 10fps masks if needed
        # If WSL did 10fps tracking (returned fps ~10), expand to full-FPS mask timeline
        track_fps = None
        try:
            track_fps = float(sam2_result.get('fps')) if isinstance(sam2_result, dict) else None
        except Exception:
            track_fps = None
        if track_fps and track_fps > 0 and abs(track_fps - 10.0) < 1.0 and total_frames > len(masks_full) >= 2:
            print(f"[SAM2] Expanding {len(masks_full)} masks (tracking @ ~10fps) to {total_frames} full-FPS frames...")
            all_masks = expand_masks_10fps(masks_full, total_frames, original_fps)
        else:
            # Fallback: strict alignment (truncate to shortest)
            if len(masks_full) != total_frames:
                min_count = min(len(masks_full), total_frames)
                masks_full = masks_full[:min_count]
                total_frames = min_count
            all_masks = masks_full

        # Fill short internal gaps
        if FILL_MASK_GAPS and all_masks:
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            max_gap = int(round(original_fps * MASK_GAP_MAX_SECONDS))
            i = 0
            while i < len(presence):
                if not presence[i]:
                    j = i
                    while j < len(presence) and not presence[j]:
                        j += 1
                    gap_len = j - i
                    if 0 < gap_len <= max_gap:
                        src_idx = i - 1 if i - 1 >= 0 and presence[i - 1] else (j if j < len(presence) else None)
                        if src_idx is not None:
                            src = all_masks[src_idx]
                            for k in range(i, j):
                                all_masks[k] = src.copy()
                    i = j
                else:
                    i += 1

        # Tail fill
        if TAIL_MASK_FILL and all_masks:
            presence = [np.sum(m > 127) > 0 for m in all_masks]
            if any(presence):
                last_idx = max(i for i, p in enumerate(presence) if p)
                if last_idx < (total_frames - 1):
                    max_fill = min(int(round(original_fps * TAIL_MASK_MAX_SECONDS)), (total_frames - 1) - last_idx)
                    if max_fill > 0:
                        src = all_masks[last_idx]
                        for k in range(1, max_fill + 1):
                            all_masks[last_idx + k] = src.copy()
                        print(f"[SAM2] Tail fill: replicated last mask across {max_fill} frame(s) to cover video end")

        # --- Validate masks have content (same check as main task) ---
        masks_with_content = sum(1 for m in all_masks if np.sum(m > 127) > 0)
        print(f"[SAM2] Mask validation: {masks_with_content}/{len(all_masks)} frames have mask content")

        if masks_with_content == 0:
            raise RuntimeError("No mask content generated - SAM2 tracking failed")

        # The rest of pipeline identical to main task from global crop → encode
        # Calculate global crop
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        for mask in all_masks:
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

        # With validation above, we should always have a valid bbox
        if max_x <= min_x or max_y <= min_y:
            raise RuntimeError("No valid bounding box found despite having mask content - unexpected state")
        global_bbox = [min_x, min_y, max_x, max_y]
        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(global_bbox, width, height, padding_ratio=0.2, min_size=128)

        # Prepare arrays (masks only - frames loaded per-segment)
        all_masks_cropped = [m[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for m in all_masks]
        global_crop = (crop_x, crop_y, crop_w, crop_h)
        print(f"[SAM2] Masks cropped, frames will be loaded per-segment")

        # --- Segment detection (FULL-FPS, boundary-safe like commit 578a2603) ---
        print(f"[{mode.upper()}] Detecting segments from FULL-FPS in-memory masks...")
        detections_per_frame = []
        for m in all_masks:
            coords = cv2.findNonZero(m)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                detections_per_frame.append((x, y, x + w, y + h))
            else:
                detections_per_frame.append(None)

        # YOLO mode: Use simpler segment detection like main branch (fewer, larger segments)
        # SAM2 mode: Use motion-based detection for fast-moving tracked objects
        if mode == 'yolo':
            # YOLO: Simple position-based detection (like server_production.py main branch)
            segments = detect_segments(detections_per_frame, position_tolerance=5, min_segment_length=10)
            if segments:
                segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
            print(f"[YOLO] Position-based detection: {len(segments)} segments (tolerance=5, min_len=10, max_gap=30)")
        elif SEGMENT_USE_MOTION_DETECTION:
            segments = detect_segments_motion_based(detections_per_frame, motion_threshold=SEGMENT_MOTION_THRESHOLD, min_segment_length=SEGMENT_MIN_LEN_FULL, max_segment_pixels=MAX_SEGMENT_PIXELS)
            print(f"[SAM2] Motion-based detection: {len(segments)} segments (threshold={SEGMENT_MOTION_THRESHOLD}px, max_pixels={MAX_SEGMENT_PIXELS:,})")
        else:
            segments = detect_segments(detections_per_frame, position_tolerance=SEGMENT_POS_TOLERANCE, min_segment_length=SEGMENT_MIN_LEN_FULL)
        if len(segments) > 1 and not SEGMENT_USE_MOTION_DETECTION and mode != 'yolo':
            segments = merge_adjacent_segments(segments, position_tolerance=SEGMENT_POS_TOLERANCE, max_gap=SEGMENT_MERGE_GAP_FULL)

        # Convert inclusive end → exclusive and clamp to total_frames
        if segments:
            segments = [(max(0, int(s)), min(int(e) + 1, total_frames), bb) for (s, e, bb) in segments if max(0, int(s)) < min(int(e) + 1, total_frames)]

            # --- BOUNDARY-SAFE: Ensure CONTIGUOUS coverage (no gaps!) ---
            # Empty-mask frames still processed but with zero mask (no inpainting)
            fixed_segments = []
            for i, (s, e, bb) in enumerate(segments):
                if i == 0 and s > 0:
                    s = 0  # First segment must start at 0
                if i > 0:
                    prev_end = fixed_segments[-1][1]
                    if s > prev_end:
                        s = prev_end  # No gaps between segments!
                if i == len(segments) - 1:
                    e = total_frames  # Last segment must reach the end
                fixed_segments.append((s, e, bb))
            segments = fixed_segments

            # Replace each segment's averaged bbox with union bbox for that segment's frames
            # This ensures fast-moving objects are fully covered within each segment
            # Also splits segments if union bbox exceeds MAX_SEGMENT_PIXELS
            if detections_per_frame:
                updated_segments = []
                for s, e, bb in segments:
                    # Compute union of all bboxes in this segment's frame range
                    segment_bboxes = [detections_per_frame[f] for f in range(s, e)
                                     if f < len(detections_per_frame) and detections_per_frame[f] is not None]
                    if segment_bboxes:
                        seg_min_x = min(b[0] for b in segment_bboxes)
                        seg_min_y = min(b[1] for b in segment_bboxes)
                        seg_max_x = max(b[2] for b in segment_bboxes)
                        seg_max_y = max(b[3] for b in segment_bboxes)
                        union_bbox = (seg_min_x, seg_min_y, seg_max_x, seg_max_y)

                        # Check if CROP (post-padding) exceeds pixel limit
                        # Use calculate_crop_region to get EXACT crop size (includes min_size, clamping)
                        _, _, crop_w, crop_h = calculate_crop_region(
                            union_bbox, width, height,
                            padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                        )
                        crop_pixels = crop_w * crop_h
                        if MAX_CROP_PIXELS > 0 and crop_pixels > MAX_CROP_PIXELS:
                            # Split this segment into smaller pieces
                            print(f"[SAM2] Segment {s}-{e}: crop {crop_pixels:,} px ({crop_w}x{crop_h}) > {MAX_CROP_PIXELS:,} limit, splitting...")
                            sub_segments = split_segment_by_pixels(s, e, detections_per_frame, MAX_CROP_PIXELS, SEGMENT_CROP_PAD_RATIO, width, height)
                            for sub_s, sub_e, sub_bb in sub_segments:
                                _, _, sub_w, sub_h = calculate_crop_region(
                                    sub_bb, width, height,
                                    padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
                                )
                                sub_crop_px = sub_w * sub_h
                                print(f"[SAM2]   -> Sub-segment {sub_s}-{sub_e} ({sub_e-sub_s}f): crop {sub_crop_px:,} px ({sub_w}x{sub_h})")
                            updated_segments.extend(sub_segments)
                        else:
                            updated_segments.append((s, e, union_bbox))
                            if union_bbox != bb:
                                print(f"[SAM2] Segment {len(updated_segments)}: union bbox {union_bbox} (was avg {bb})")
                    else:
                        updated_segments.append((s, e, bb))
                segments = updated_segments
        else:
            # No segments detected - cover entire video
            segments = [(0, total_frames, global_bbox)]

        # Split segments that exceed MAX_SEGMENT_FRAMES to prevent OOM
        if MAX_SEGMENT_FRAMES > 0:
            split_segments = []
            for s, e, bb in segments:
                seg_len = e - s
                if seg_len > MAX_SEGMENT_FRAMES:
                    # Split into chunks of MAX_SEGMENT_FRAMES
                    num_chunks = (seg_len + MAX_SEGMENT_FRAMES - 1) // MAX_SEGMENT_FRAMES
                    chunk_size = seg_len // num_chunks
                    for i in range(num_chunks):
                        chunk_start = s + i * chunk_size
                        chunk_end = s + (i + 1) * chunk_size if i < num_chunks - 1 else e
                        split_segments.append((chunk_start, chunk_end, bb))
                    print(f"[SAM2] Split {seg_len}f segment into {num_chunks} chunks of ~{chunk_size}f")
                else:
                    split_segments.append((s, e, bb))
            segments = split_segments

        # --- POST-PROCESS: Remove empty mask frames and split segments ---
        MIN_FRAMES_SPLIT = 5  # Min frames for segments broken by empty masks

        def filter_empty_frames_from_segments(segs, masks, det_per_frame):
            """Remove empty mask frames, split if empty in middle."""
            filtered = []
            for s, e, bb in segs:
                # Find runs of non-empty frames
                runs = []
                run_start = None
                for f in range(s, e):
                    has_content = f < len(masks) and np.sum(masks[f] > 127) > 0
                    if has_content:
                        if run_start is None:
                            run_start = f
                    else:
                        if run_start is not None:
                            runs.append((run_start, f))
                            run_start = None
                if run_start is not None:
                    runs.append((run_start, e))

                # Create segments from runs
                is_split = len(runs) > 1
                min_len = MIN_FRAMES_SPLIT if is_split else SEGMENT_MIN_LEN_FULL
                for run_s, run_e in runs:
                    if run_e - run_s >= min_len:
                        # Compute bbox for this run
                        run_bboxes = [det_per_frame[f] for f in range(run_s, run_e)
                                      if f < len(det_per_frame) and det_per_frame[f]]
                        if run_bboxes:
                            run_bb = (min(b[0] for b in run_bboxes), min(b[1] for b in run_bboxes),
                                      max(b[2] for b in run_bboxes), max(b[3] for b in run_bboxes))
                        else:
                            run_bb = bb
                        filtered.append((run_s, run_e, run_bb))
                        if is_split:
                            print(f"[SAM2] Empty-mask split: frames {run_s}-{run_e-1} ({run_e-run_s}f)")
            return filtered

        orig_count = len(segments)
        segments = filter_empty_frames_from_segments(segments, all_masks, detections_per_frame)
        if len(segments) != orig_count:
            print(f"[SAM2] Filtered empty masks: {orig_count} -> {len(segments)} segments")

        print(f"[SAM2] Detected {len(segments)} segments (FULL-FPS, boundary-safe)")
        for idx, (start, end, bbox) in enumerate(segments):
            print(f"[SAM2]   Segment {idx+1}: frames {start}-{end-1} ({end-start}f) bbox={bbox}")

        # Now reuse the same merging + ProPainter logic as the main task
        # For brevity, call back into process_sam2_interactive_task's internal pipeline is not trivial; so inline minimal subset
        # We re-use the same process_segment closure pattern

        # Settings
        import torch
        use_fp16 = torch.cuda.is_available()
        faster_propainter_pipeline = get_propainter_pipeline()
        output_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_output")
        # Don't create output_dir - not needed with return_frames=True

        # Track segment metadata (NOT frames - those stay in VRAM!)
        segment_results = {}
        processed_frames = set()  # Track which frames have been processed

        # REVOLUTIONARY: Store all frames in VRAM with YUV420 compression (8:1 ratio!)
        vram_frames = {}  # {frame_idx: compressed_yuv420_dict}

        # PRE-LOAD ALL FRAMES ONCE (workers will share!)
        print(f"[SAM2] Pre-loading {total_frames} frames to shared VRAM (YUV420)...")
        preloaded_frames = {}  # {frame_idx: compressed_yuv420}
        cap_preload = cv2.VideoCapture(video_path)
        for frame_idx in range(total_frames):
            ret, frame = cap_preload.read()
            if ret:
                preloaded_frames[frame_idx] = VRAMCompressor.compress_frame(frame)
        cap_preload.release()
        print(f"[SAM2] All {len(preloaded_frames)} frames pre-loaded - workers will SHARE!")

        def process_segment_local(seg_idx, start_f, end_f, seg_bbox):
            duration = end_f - start_f
            log_end = end_f - 1 if end_f > start_f else end_f
            # Calculate segment-specific crop directly from seg_bbox (in original frame coords)
            seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                list(seg_bbox),  # Use segment bbox directly in frame coordinates
                width, height, padding_ratio=SEGMENT_CROP_PAD_RATIO, min_size=128
            )
            # Hasan branch used FIXED values for speed (not resolution-based)
            subvideo_length = 120  # Hasan: 120 fixed (larger batch = faster)
            neighbor_length = 10   # Hasan: 10 (more temporal context)
            ref_stride = 10        # Hasan: 10 (not 15)
            pad_left = min(SEGMENT_TEMPORAL_PAD, start_f)
            pad_right = min(SEGMENT_TEMPORAL_PAD, total_frames - end_f)
            proc_start = start_f - pad_left
            proc_end = end_f + pad_right

            print(f"[SAM2] Segment {seg_idx+1}: frames {start_f}-{log_end} ({duration}f), crop={seg_crop_w}x{seg_crop_h}, pad=+/-{SEGMENT_TEMPORAL_PAD} => proc {proc_start}-{proc_end-1}")

            # GET FRAMES FROM PRE-LOADED SHARED VRAM (ZERO disk I/O!)
            print(f"[FAST] Using {proc_end - proc_start} frames from memory (ZERO disk I/O!)")
            seg_frames = []
            seg_masks = []
            for frame_idx in range(proc_start, proc_end):
                # Decompress from shared preloaded frames
                full_frame = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                # Apply segment-specific crop directly (seg_crop coords are in frame coords)
                seg_frame = full_frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                seg_frames.append(np.ascontiguousarray(seg_frame))
                # Crop mask from original all_masks (not pre-cropped)
                if frame_idx < len(all_masks):
                    seg_mask = all_masks[frame_idx][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                    seg_masks.append(np.ascontiguousarray(seg_mask))
            if not seg_frames or not seg_masks:
                print(f"[SAM2] WARNING: Segment {seg_idx+1} has no frames/masks! seg_frames={len(seg_frames)}, seg_masks={len(seg_masks)}")
                return seg_idx, None, None, (proc_start, proc_end)

            # Run ProPainter - ZERO DISK I/O with return_frames=True!
            output_frames = faster_propainter_pipeline(
                video='dummy', mask='dummy', output=output_dir,
                resize_ratio=1.0, mask_dilation=SAM2_MASK_DILATION,
                ref_stride=ref_stride, neighbor_length=neighbor_length, subvideo_length=subvideo_length,
                raft_iter=10, mode="video_inpainting", save_fps=int(original_fps), save_frames=False,
                fp16=use_fp16, use_cached_models=True, frames_array=seg_frames, masks_array=seg_masks,
                return_frames=True  # ZERO DISK I/O!
            )

            expected_count = proc_end - proc_start
            if not output_frames:
                print(f"[SAM2] WARNING: Segment {seg_idx+1} returned no frames!")
                return seg_idx, None, None, (proc_start, proc_end)
            elif len(output_frames) != expected_count:
                print(f"[SAM2] WARNING: Segment {seg_idx+1} frame count mismatch! Expected {expected_count}, got {len(output_frames)}")
            else:
                print(f"[SAM2] Segment {seg_idx+1}: ProPainter returned {len(output_frames)} frames (OK)")

            # REVOLUTIONARY: Store in VRAM with YUV420 compression (8:1 ratio, ZERO disk I/O!)
            # Only save CORE frames (start_f to end_f), skip padding frames
            core_count = end_f - start_f
            print(f"[SAM2] Segment {seg_idx+1}: storing {core_count} core frames in VRAM (skip {len(output_frames) - core_count} padding)...")
            frames_stored = 0
            for i in range(len(output_frames)):
                frame_idx = proc_start + i
                # Skip padding frames - only save core frames [start_f, end_f)
                if frame_idx < start_f or frame_idx >= end_f:
                    continue
                # GET from shared preloaded frames (ZERO disk I/O!)
                orig_frame = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
                # Composite segment output onto original frame (seg_crop coords are in frame coords)
                orig_frame[seg_crop_y:seg_crop_y + seg_crop_h,
                          seg_crop_x:seg_crop_x + seg_crop_w] = output_frames[i]
                # Store in VRAM with YUV420 compression (8:1 ratio!)
                vram_frames[frame_idx] = VRAMCompressor.compress_frame(orig_frame)
                processed_frames.add(frame_idx)
                frames_stored += 1
            del output_frames  # Free memory immediately!
            print(f"[SAM2] Segment {seg_idx+1}: stored {frames_stored} core frames in VRAM")

            # Return success flag and metadata (NOT frames!)
            return seg_idx, True, (seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h), (proc_start, proc_end)

        # Process segments (parallel when enabled, like main task)
        segments_to_process = segments or [(0, total_frames, [0, 0, width, height])]

        # Use mode-specific worker count (YOLO: 4, SAM2: 2)
        segment_workers = 4 if mode == 'yolo' else SEGMENT_WORKERS

        if SAM2_PARALLEL_SEGMENTS and segment_workers > 1 and len(segments_to_process) > 1:
            print(f"[{mode.upper()}] Processing {len(segments_to_process)} segments in PARALLEL (workers={min(segment_workers, len(segments_to_process))})...")

            max_workers = min(segment_workers, len(segments_to_process))
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments_to_process):
                    futures[executor.submit(process_segment_local, seg_idx, start_f, end_f, seg_bbox)] = seg_idx

                completed = 0
                for future in as_completed(futures):
                    seg_idx_done = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"[ERROR] Segment {seg_idx_done} raised: {e}")
                        import traceback
                        traceback.print_exc()
                        result = (seg_idx_done, None, None, (0, 0))
                    segment_results[seg_idx_done] = result
                    completed += 1
                    print(f"[SAM2] Completed {completed}/{len(segments_to_process)} segments")
        else:
            print(f"[SAM2] Processing {len(segments_to_process)} segment(s) sequentially...")
            for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments_to_process):
                segment_results[seg_idx] = process_segment_local(seg_idx, start_f, end_f, seg_bbox)

        # --- REVOLUTIONARY: Direct VRAM → FFmpeg pipe → NVENC (ZERO disk I/O!) ---
        print(f"[SAM2] Segment processing complete - {len(vram_frames)} frames stored in VRAM")
        print(f"[SAM2] Encoding directly from VRAM via FFmpeg pipe (ZERO disk I/O!)...")
        import time as _time
        _encode_start = _time.time()

        output_path = os.path.join(RESULT_DIR, f"{video_id}_sam2_removed.mp4")

        # Get frame dimensions from first available frame
        first_frame = VRAMCompressor.decompress_frame(
            vram_frames[min(vram_frames.keys())] if vram_frames else preloaded_frames[0]
        )
        frame_height, frame_width = first_frame.shape[:2]

        # FFmpeg command: pipe rawvideo → NVENC → MP4 with audio from original
        ffmpeg_cmd = [
            str(FFMPEG_EXE), '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{frame_width}x{frame_height}',
            '-r', str(int(original_fps)),
            '-i', 'pipe:0',  # Video from stdin
            '-i', video_path,  # Audio from original
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'h264_nvenc',  # GPU encoding!
            '-preset', 'p1',  # Fastest NVENC preset
            '-b:v', '8M',
            '-bufsize', '16M',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]

        # Start FFmpeg process
        # NOTE: Use DEVNULL for stdout/stderr to prevent pipe deadlock on long videos!
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Stream ALL frames directly to FFmpeg (processed + unprocessed)
        for frame_idx in range(total_frames):
            # Progress logging every 100 frames
            if frame_idx % 100 == 0:
                print(f"[SAM2] Encoding frame {frame_idx}/{total_frames}...")

            if frame_idx in vram_frames:
                # Processed frame from segment
                frame_bgr = VRAMCompressor.decompress_frame(vram_frames[frame_idx])
            elif frame_idx in preloaded_frames:
                # Unprocessed frame from preloaded
                frame_bgr = VRAMCompressor.decompress_frame(preloaded_frames[frame_idx])
            else:
                print(f"[WARNING] Frame {frame_idx} not found in VRAM, skipping")
                continue

            proc.stdin.write(frame_bgr.tobytes())

        proc.stdin.close()
        proc.wait(timeout=300)

        if proc.returncode != 0:
            print(f"[ERROR] FFmpeg NVENC encoding failed (code {proc.returncode})")
            raise RuntimeError(f"Video encoding failed (code {proc.returncode})")

        _encode_time = _time.time() - _encode_start
        print(f"[SAM2] NVENC encode complete: {total_frames} frames in {_encode_time:.2f}s ({total_frames/_encode_time:.1f} fps)")

        # Free VRAM
        import torch as _torch
        del vram_frames
        _torch.cuda.empty_cache()

        # Upload to B2 + Cloudflare CDN (inlined - no heavy imports!)
        cdn_url = None
        try:
            from b2sdk.v2 import B2Api, InMemoryAccountInfo
            import time as _upload_time

            B2_KEY_ID = os.getenv('B2_KEY_ID', '00539db5c1104b50000000002')
            B2_APP_KEY = os.getenv('B2_APP_KEY', 'K005HJKUP7ahSNJ1wgQHDDJ+uEATiU4')
            B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
            B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

            if os.getenv('B2_UPLOAD_ENABLED', '1') == '1':
                timestamp = int(_upload_time.time())
                remote_path = f"results/{timestamp}_{os.path.basename(output_path)}"

                print(f"[B2] Uploading to {B2_BUCKET}/{remote_path}...")
                _b2_start = _upload_time.time()
                info = InMemoryAccountInfo()
                b2_api = B2Api(info)
                b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
                bucket = b2_api.get_bucket_by_name(B2_BUCKET)
                bucket.upload_local_file(local_file=output_path, file_name=remote_path)
                cdn_url = f"{B2_CDN_URL}/{remote_path}"
                _b2_time = _upload_time.time() - _b2_start
                print(f"[B2] Upload complete in {_b2_time:.1f}s - CDN URL: {cdn_url}")
        except ImportError:
            print(f"[B2] b2sdk not installed - skipping upload")
        except Exception as e:
            print(f"[B2] Upload failed: {e}")

        # Cleanup temp files on success
        print(f"[SAM2] Cleaning up...")
        import shutil as _shutil
        for temp_path in [output_dir, masks_dir]:
            if isinstance(temp_path, str) and os.path.exists(temp_path):
                if os.path.isdir(temp_path):
                    _shutil.rmtree(temp_path)

        return {
            'status': 'success', 'video_id': video_id,
            'video_path': video_path, 'output_path': cdn_url or output_path,
            'total_frames': total_frames, 'masks_generated': len(all_masks), 'masks_expanded': len(all_masks),
            'segments_processed': len(segments or []), 'width': width, 'height': height,
            'fps': original_fps, 'pipeline': 'full_fps_segments', 'message': f'SAM2 pipeline complete! Processed {len(segments or [])} segment(s)'
        }
    except Exception as e:
        print(f"[ERROR] Continue-after-masks task failed: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup temp files on failure
        try:
            import shutil as _shutil
            for temp_path in [output_dir, masks_dir]:
                if isinstance(temp_path, str) and os.path.exists(temp_path):
                    if os.path.isdir(temp_path):
                        _shutil.rmtree(temp_path)
                        print(f"[CLEANUP] Removed {temp_path}")
        except Exception as cleanup_err:
            print(f"[CLEANUP] Warning: {cleanup_err}")
        raise


#============================================================================
# YOLO MODE CELERY TASKS
# Fast parallel processing with Celery chord pattern (like server_production.py)
#============================================================================

from celery import chord

@celery.task(bind=True, name='watermark.yolo_prepare_video', queue='yolo')
def yolo_prepare_video_task(self, video_path, api_base=None, video_id=None):
    """
    YOLO Mode Phase 1: Prepare video for distributed processing
    - Run YOLO detection on all frames
    - Generate masks with GPU batch processing
    - Detect segments (handles moving watermarks)
    - Dispatch segment tasks via Celery chord
    """
    try:
        import json
        import time as _time
        import shutil

        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Preparing video (YOLO mode)'})

        detector = get_yolo_detector()
        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing")

        # Download video if needed
        from pathlib import PureWindowsPath, Path
        base_name = PureWindowsPath(video_path).name if '\\' in video_path else os.path.basename(video_path)
        UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
        local_video_path = os.path.join(UPLOAD_DIR, base_name)

        if os.path.exists(local_video_path):
            print(f"[YOLO] Video already exists locally: {local_video_path}")
            video_path = local_video_path
        elif not os.path.exists(video_path):
            tunnel = api_base or os.getenv('TUNNEL_URL')
            if tunnel:
                import requests
                from urllib.parse import urljoin
                download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                print(f"[YOLO] Downloading video: {download_url}")
                r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                r.raise_for_status()
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(local_video_path, 'wb') as f:
                    f.write(r.content)
                video_path = local_video_path
            else:
                raise Exception(f"Video not found: {video_path}")

        print(f"[YOLO] Preparing video: {video_path}")
        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Analyzing video'})

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        base_name = Path(video_path).stem
        video_id = video_id or uuid.uuid4().hex[:8]

        print(f"[YOLO] Video: {width}x{height} @ {fps} fps ({total_frames} frames)")

        # Store video metadata in Redis
        redis_client = celery.backend.client
        redis_client.set(f"yolo_video:{video_id}:base_name", base_name)
        redis_client.set(f"yolo_video:{video_id}:video_path", video_path)

        # Load all frames for batch detection
        print(f"[YOLO] Loading {total_frames} frames for batch detection...")
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Loading frames'})

        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()
        frames_processed = len(all_frames)

        # Batch YOLO detection (748 fps on RTX 4090!)
        print(f"[YOLO] Running batch detection on {frames_processed} frames...")
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'Detecting watermarks (batch)'})

        batch_start = _time.time()
        all_detections = detector.detect_batch(all_frames, confidence_threshold=0.15, padding=0, batch_size=64)
        batch_duration = _time.time() - batch_start
        print(f"[YOLO] Batch detection: {batch_duration:.3f}s ({frames_processed/batch_duration:.1f} fps)")

        # Process detections
        zero_mask = np.zeros((height, width), dtype=np.uint8)
        bboxes_per_frame = []
        frames_with_watermark = 0
        last_valid_bbox = None
        all_masks = []

        for i, detections in enumerate(all_detections):
            if detections:
                frames_with_watermark += 1
                last_valid_bbox = detections[0]['bbox']
                bboxes_per_frame.append(last_valid_bbox)
            elif last_valid_bbox:
                bboxes_per_frame.append(last_valid_bbox)
            else:
                bboxes_per_frame.append(None)

        # Create masks (GPU batch if available)
        print(f"[YOLO] Creating {frames_processed} masks...")
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': 'Creating masks'})

        if hasattr(detector, 'use_gpu_masks') and detector.use_gpu_masks:
            all_masks = detector.create_masks_batch_gpu(all_frames, all_detections)
        else:
            all_masks = [
                detector.create_mask(frame, dets) if dets else zero_mask
                for frame, dets in zip(all_frames, all_detections)
            ]

        print(f"[YOLO] Detection complete: {frames_with_watermark}/{frames_processed} frames with watermarks")

        # Detect segments
        segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
        if segments:
            segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
            print(f"[YOLO] Detected {len(segments)} segments")
        else:
            segments = [(0, frames_processed-1, last_valid_bbox if last_valid_bbox else [0,0,width,height])]
            print("[YOLO] No segments detected - processing entire video as one segment")

        # Store frames and masks in shared directory for segment workers
        shared_frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_frames")
        shared_mask_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_masks")
        shared_cleaned_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_cleaned")
        os.makedirs(shared_frames_dir, exist_ok=True)
        os.makedirs(shared_mask_dir, exist_ok=True)
        os.makedirs(shared_cleaned_dir, exist_ok=True)

        print(f"[YOLO] Writing {frames_processed} frames and masks to disk...")
        self.update_state(state='PROCESSING', meta={'progress': 40, 'status': 'Saving frames/masks'})

        for i in range(frames_processed):
            cv2.imwrite(os.path.join(shared_frames_dir, f"{i:04d}.png"), all_frames[i])
            cv2.imwrite(os.path.join(shared_mask_dir, f"{i:04d}.png"), all_masks[i])

        # Free memory
        del all_frames
        del all_masks

        # Prepare segment task data
        segment_tasks_data = []
        api_base_url = api_base or os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL')

        for seg_idx, (start_frame, end_frame, bbox) in enumerate(segments):
            segment_tasks_data.append({
                'seg_idx': seg_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'bbox': list(bbox) if bbox else [0,0,width,height],
                'video_id': video_id,
                'base_name': base_name,
                'width': width,
                'height': height,
                'fps': fps,
                'video_path': video_path,
                'shared_frames_dir': shared_frames_dir,
                'shared_mask_dir': shared_mask_dir,
                'shared_cleaned_dir': shared_cleaned_dir,
                'total_segments': len(segments),
            })

        print(f"[YOLO] Dispatching {len(segments)} segment tasks via chord...")
        self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Dispatching {len(segments)} parallel tasks'})

        # Store tracking info
        redis_client.set(f"yolo_segments:{video_id}:total", len(segments))

        # Create chord: segments in parallel, finalize when all complete
        segment_sigs = [yolo_process_segment_task.s(seg_data) for seg_data in segment_tasks_data]
        prepare_result = {
            'video_id': video_id,
            'base_name': base_name,
            'video_path': video_path,
            'total_segments': len(segments),
            'width': width,
            'height': height,
            'fps': fps,
            'shared_cleaned_dir': shared_cleaned_dir,
        }

        workflow = chord(segment_sigs)(yolo_finalize_video_task.s(prepare_result=prepare_result))

        print(f"[YOLO] Chord dispatched! Finalize callback ID: {workflow.id}")

        return {
            'chord_id': f'yolo_distributed_{video_id}',
            'status': 'processing',
            'message': f'Chord workflow: {len(segments)} segments -> finalize callback',
            'video_id': video_id,
        }

    except Exception as e:
        print(f"[YOLO ERROR] Prepare task failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.yolo_process_segment', queue='yolo')
def yolo_process_segment_task(self, segment_data):
    """
    YOLO Mode Phase 2: Process one segment
    - Load frames and masks from shared storage
    - Run ProPainter inpainting
    - Write cleaned frames to shared cleaned directory
    - Signal background encoder via Redis pub/sub
    """
    try:
        import json
        import time as _time

        seg_idx = segment_data['seg_idx']
        total_segments = segment_data['total_segments']
        start_frame = segment_data['start_frame']
        end_frame = segment_data['end_frame']
        bbox = segment_data['bbox']
        video_id = segment_data['video_id']
        base_name = segment_data['base_name']
        width = segment_data['width']
        height = segment_data['height']
        fps = segment_data['fps']
        shared_frames_dir = segment_data['shared_frames_dir']
        shared_mask_dir = segment_data['shared_mask_dir']
        shared_cleaned_dir = segment_data['shared_cleaned_dir']

        print(f"\n[YOLO SEGMENT] Processing segment {seg_idx+1}/{total_segments}: frames {start_frame}-{end_frame}")
        self.update_state(state='STARTED', meta={'progress': 0, 'status': f'Processing segment {seg_idx+1}'})

        seg_start = _time.time()

        # Calculate crop region
        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"   [CROP] Crop region: {crop_w}x{crop_h} @ ({crop_x},{crop_y})")

        # Temporal padding for ProPainter context
        neighbor_padding = 5
        padded_start = max(0, start_frame - neighbor_padding)
        padded_end = end_frame + neighbor_padding

        # Load frames and masks with padding
        print(f"   [LOAD] Loading frames {padded_start}-{padded_end}...")
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Loading frames'})

        seg_frames = []
        seg_masks = []
        full_frames = []  # Keep full frames for paste-back

        for frame_idx in range(padded_start, padded_end + 1):
            frame_path = os.path.join(shared_frames_dir, f"{frame_idx:04d}.png")
            mask_path = os.path.join(shared_mask_dir, f"{frame_idx:04d}.png")

            if os.path.exists(frame_path):
                full_frame = cv2.imread(frame_path)
                if full_frame is not None:
                    # Crop frame for ProPainter
                    cropped = full_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    seg_frames.append(np.ascontiguousarray(cropped))

                    # Keep full frame only for core segment (paste-back)
                    if start_frame <= frame_idx <= end_frame:
                        full_frames.append((frame_idx, full_frame))

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    seg_masks.append(np.ascontiguousarray(cropped_mask))

        if not seg_frames or not seg_masks:
            raise RuntimeError(f"No frames/masks loaded for segment {seg_idx}")

        print(f"   [OK] Loaded {len(seg_frames)} frames, {len(seg_masks)} masks")

        # Run ProPainter
        print(f"   [PROPAINTER] Running inpainting on {len(seg_frames)} frames...")
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': 'Running ProPainter'})

        pipeline = get_propainter_pipeline()
        use_fp16 = torch.cuda.is_available()

        output_frames = pipeline(
            video='dummy',
            mask='dummy',
            output=TEMP_DIR,
            resize_ratio=1.0,
            mask_dilation=SAM2_MASK_DILATION,
            ref_stride=10,
            neighbor_length=10,
            subvideo_length=120,
            raft_iter=10,
            mode="video_inpainting",
            save_fps=fps,
            save_frames=False,
            fp16=use_fp16,
            use_cached_models=True,
            frames_array=seg_frames,
            masks_array=seg_masks,
            return_frames=True
        )

        if not output_frames:
            raise RuntimeError(f"ProPainter returned no frames for segment {seg_idx}")

        print(f"   [OK] ProPainter returned {len(output_frames)} frames")

        # Composite and save to shared cleaned directory
        print(f"   [COMPOSITE] Pasting results onto original frames...")
        self.update_state(state='PROCESSING', meta={'progress': 70, 'status': 'Compositing frames'})

        frames_saved = 0
        paste_idx = 0

        for i in range(len(output_frames)):
            frame_idx = padded_start + i
            # Only save core frames (skip padding)
            if frame_idx < start_frame or frame_idx > end_frame:
                continue

            if paste_idx < len(full_frames):
                saved_frame_idx, orig_frame = full_frames[paste_idx]
                paste_idx += 1

                # Paste cropped result onto original frame
                orig_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = output_frames[i]

                # Save to shared cleaned directory (global frame index)
                output_path = os.path.join(shared_cleaned_dir, f"{frame_idx:04d}.png")
                cv2.imwrite(output_path, orig_frame)
                frames_saved += 1

        print(f"   [OK] Saved {frames_saved} cleaned frames to {shared_cleaned_dir}")

        # Free memory
        del output_frames
        del seg_frames
        del seg_masks
        del full_frames
        clear_gpu_memory()

        seg_duration = _time.time() - seg_start
        print(f"   [DONE] Segment {seg_idx+1} complete in {seg_duration:.2f}s")

        # Store segment metadata in Redis for background encoder
        redis_client = celery.backend.client
        segment_key = f"yolo_video:{video_id}:segment:{seg_idx}"
        redis_client.hset(segment_key, mapping={
            'cleaned_dir': shared_cleaned_dir,
            'fps': str(fps),
            'frame_count': str(frames_saved),
            'base_name': base_name,
            'start_frame': str(start_frame),
            'end_frame': str(end_frame),
            'status': 'processed',
        })

        # Signal background encoder that this segment is ready
        signal_data = json.dumps({
            'video_id': video_id,
            'seg_idx': seg_idx,
            'total_segments': total_segments,
        })
        redis_client.publish('yolo_segment_ready', signal_data)
        print(f"   [SIGNAL] Published yolo_segment_ready for segment {seg_idx}")

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': f'Segment {seg_idx+1} complete'})

        return {
            'status': 'success',
            'seg_idx': seg_idx,
            'frames_processed': frames_saved,
            'duration': seg_duration,
        }

    except Exception as e:
        print(f"[YOLO SEGMENT ERROR] Segment {segment_data.get('seg_idx', '?')} failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.yolo_finalize_video', queue='yolo')
def yolo_finalize_video_task(self, segment_results, prepare_result):
    """
    YOLO Mode Phase 3: Finalize video
    Returns immediately - actual encoding/finalization happens in background encoder thread
    """
    try:
        video_id = prepare_result['video_id']
        base_name = prepare_result['base_name']
        total_segments = prepare_result['total_segments']

        print(f"\n[YOLO FINALIZE TASK] All segments complete! Background encoder handling finalization for: {base_name}")
        print(f"[YOLO FINALIZE TASK] Encoding + concatenation happening asynchronously in background")

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Background encoding in progress'})

        return {
            'status': 'background_encoding',
            'message': f'Segments complete. Encoding + finalization in progress (background thread)',
            'video_id': video_id,
            'total_segments': total_segments,
            'check_status_key': f'yolo_video:{video_id}:status',
            'final_path_key': f'yolo_video:{video_id}:final_path',
        }

    except Exception as e:
        print(f"[YOLO FINALIZE ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        raise
