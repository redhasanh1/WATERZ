"""
Production Server for Watermark Removal SaaS
- Async queue processing with Celery + Redis
- GPU-optimized YOLO detection + ProPainter inpainting
- Keeps your PC usable while serving customers
- Designed for $1Mi/month scale
- ALL FILES STAY ON D DRIVE (inside watermarkz folder)
"""

import sys
import os
import importlib
import shutil
from pathlib import Path

# CRITICAL: Force ALL temp/cache to D drive (watermarkz folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
UPLOAD_DIR = os.path.join(SCRIPT_DIR, 'uploads')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')
DEBUG_DIR = os.path.join(RESULT_DIR, 'debug_masks')
PYTHON_PACKAGES_DIR = os.path.join(SCRIPT_DIR, 'python_packages')
PROPAINTER_SCRIPT = os.path.join(SCRIPT_DIR, 'ProPainter', 'inference_propainter.py')
PROPAINTER_OUTPUT_ROOT = os.path.join(RESULT_DIR, 'propainter')
PROPAINTER_MASK_ROOT = os.path.join(TEMP_DIR, 'propainter_masks')
PROPAINTER_FLOW_BACKEND = os.getenv('PROPAINTER_FLOW_BACKEND', 'raft')  # 'raft' or 'fastflownet'
try:
    _segment_workers_env = int(os.getenv('SEGMENT_WORKERS', '2'))
except ValueError:
    _segment_workers_env = 2
SEGMENT_WORKERS = max(1, _segment_workers_env)

# Force parallel segmentation for multi-GPU/multi-worker distribution
os.environ.setdefault('MIN_SEGMENTS', '4')  # Split videos into at least 4 segments for parallel processing
os.environ.setdefault('MIN_CHUNK_FRAMES', '30')  # Minimum frames per segment

# Create directories
for directory in [TEMP_DIR, CACHE_DIR, UPLOAD_DIR, RESULT_DIR, DEBUG_DIR, PROPAINTER_OUTPUT_ROOT, PROPAINTER_MASK_ROOT]:
    os.makedirs(directory, exist_ok=True)

# Override ALL temp/cache environment variables
os.environ['TEMP'] = TEMP_DIR
os.environ['TMP'] = TEMP_DIR
os.environ['TMPDIR'] = TEMP_DIR
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['PIP_CACHE_DIR'] = os.path.join(SCRIPT_DIR, 'pip_cache')
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
os.environ['OPENCV_TEMP_PATH'] = TEMP_DIR


def _ensure_cuda_torch():
    """
    Make sure we end up with a CUDA-enabled torch build.
    Prefer the system install if it has CUDA; otherwise fall back to python_packages.
    """
    def _import_torch(disable_triton_retry: bool = False):
        """
        Import torch and, if we hit the duplicated TORCH_LIBRARY Triton error,
        retry exactly once with Triton disabled.
        """
        try:
            return importlib.import_module('torch')
        except RuntimeError as exc:
            message = str(exc)
            if (
                not disable_triton_retry
                and "Only a single TORCH_LIBRARY" in message
            ):
                os.environ['PYTORCH_DISABLE_TRITON'] = '1'
                for module_name in ['torch', 'triton', 'torch_triton', 'torchvision._cuda', 'torchvision._C']:
                    sys.modules.pop(module_name, None)
                importlib.invalidate_caches()
                return _import_torch(disable_triton_retry=True)
            raise

    try:
        _torch_test = _import_torch()
        if hasattr(_torch_test, 'cuda') and _torch_test.cuda.is_available():
            sys.modules['torch'] = _torch_test
            return
        # GPU not available – continue with the existing torch module (CPU fallback)
        sys.modules['torch'] = _torch_test
        print("⚠️  CUDA not detected in system torch; continuing with CPU mode.")
        return
    except Exception:
        # Drop whatever was imported and fall back to bundled packages
        sys.modules.pop('torch', None)
        sys.modules.pop('triton', None)

    if PYTHON_PACKAGES_DIR not in sys.path:
        sys.path.insert(0, PYTHON_PACKAGES_DIR)

    torch_cuda = _import_torch()
    if not hasattr(torch_cuda, 'cuda') or not torch_cuda.cuda.is_available():
        raise RuntimeError(
            "CUDA-enabled torch not available in python_packages. "
            "Reinstall dependencies or run INSTALL_WAVEMIX.bat."
        )
    sys.modules['torch'] = torch_cuda


_ensure_cuda_torch()

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from celery import Celery
import cv2
import numpy as np
import io
import time
import hashlib
import uuid
from datetime import datetime
import redis
import threading
import secrets
import hmac
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import faster-propainter modules for direct pipeline processing
# Note: These imports are moved to inside functions to avoid startup errors
# sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'faster-propainter-main'))
# from watermark import pipeline as faster_propainter_pipeline
# from mytimer import timer_decorator  
# from pre_post_process import crop_video_mask, merge_videos_with_mask

app = Flask(__name__, static_folder='web')
# CORS: allow cross-origin calls to /api/* and accept the ngrok header if present
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
    expose_headers=["Content-Disposition"]
)

# ----------------------------------------------------------------------------
# Simple access logging (Waitress doesn't emit per‑request access logs by default)
# ----------------------------------------------------------------------------
import sys
import logging

# Toggle per-request access logs with ACCESS_LOGS=1 (default disabled)
ENABLE_ACCESS_LOGS = str(os.getenv('ACCESS_LOGS', '0')).lower() in ('1', 'true', 'yes', 'on')
LOG_LEVEL = logging.INFO if ENABLE_ACCESS_LOGS else logging.WARNING
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL)

from flask import g

@app.before_request
def _log_request_start():
    if not ENABLE_ACCESS_LOGS:
        return
    try:
        g._req_start = time.time()
        g._req_id = secrets.token_hex(4)
        ip = request.headers.get('X-Forwarded-For', request.remote_addr) or '-'
        qs = request.query_string.decode('utf-8', errors='ignore')
        path_qs = request.path + (('?' + qs) if qs else '')
        logging.info(f"--> {g._req_id} {request.method} {path_qs} from {ip}")
    except Exception:
        pass

@app.after_request
def _log_request_end(response):
    if ENABLE_ACCESS_LOGS:
        try:
            rid = getattr(g, '_req_id', '-')
            dur_ms = int((time.time() - getattr(g, '_req_start', time.time())) * 1000)
            length = response.calculate_content_length() or 0
            logging.info(f"<-- {rid} {response.status_code} {length}b {dur_ms}ms {request.method} {request.path}")
        except Exception:
            pass
    return response

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return ('', 204)
    """Basic health endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'message': 'Flask API server running - workers handle processing'
    })

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    # Prevent MIME sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    # XSS protection
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    # Content Security Policy
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://pagead2.googlesyndication.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self' https:; frame-src https://pagead2.googlesyndication.com;"
    # Remove server header
    response.headers.pop('Server', None)
    return response

# ============================================================================
# Configuration - ALL ON D DRIVE
# ============================================================================

# Security - Generate secret key for session encryption
SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SECRET_KEY'] = SECRET_KEY

# Redis configuration (for queue + caching) - AUTO-LOADED from ngrok
# Read from redis_url.txt (auto-generated by START_ALL.bat from ngrok API)
REDIS_URL = None
# SCRIPT_DIR is already defined above as os.path.dirname(os.path.abspath(__file__))
redis_url_file = os.path.join(os.path.dirname(SCRIPT_DIR), 'redis_url.txt')
if os.path.exists(redis_url_file):
    with open(redis_url_file, 'r') as f:
        REDIS_URL = f.read().strip()
    print(f"✅ Auto-loaded Redis URL from {redis_url_file}")
    print(f"   Using: {REDIS_URL}")
else:
    # Fallback to environment variable only if file doesn't exist
    REDIS_URL = os.getenv('REDIS_URL', 'redis://:watermarkz_secure_2024@localhost:6379/0')
    print(f"⚠️ WARNING: redis_url.txt not found at {redis_url_file}")
    if REDIS_URL == 'redis://:watermarkz_secure_2024@localhost:6379/0':
        print("   Using localhost Redis - run START_ALL.bat to use ngrok tunnel!")
    else:
        print(f"   Using REDIS_URL from environment: {REDIS_URL}")

app.config['broker_url'] = REDIS_URL
app.config['result_backend'] = REDIS_URL  # Store results in Redis
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR  # D drive only!
app.config['TEMP_FOLDER'] = TEMP_DIR  # D drive only!

# Rate limiting - prevent abuse
UPLOAD_RATE_LIMIT = {}  # IP -> (count, timestamp)

# Input validation
def sanitize_filename(filename):
    """Remove dangerous characters from filenames"""
    import re
    # Remove path traversal attempts
    filename = os.path.basename(filename)
    # Only allow alphanumeric, dots, dashes, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename

def validate_url(url):
    """Validate and sanitize URLs to prevent SSRF attacks"""
    from urllib.parse import urlparse

    if not url or not isinstance(url, str):
        return False

    # Basic length check
    if len(url) > 2048:
        return False

    try:
        parsed = urlparse(url)

        # Must have scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False

        # Only allow http/https
        if parsed.scheme not in ['http', 'https']:
            return False

        # Block localhost/internal IPs to prevent SSRF
        blocked_hosts = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
        if any(blocked in parsed.netloc.lower() for blocked in blocked_hosts):
            return False

        # Block private IP ranges
        import ipaddress
        try:
            # Extract hostname without port
            hostname = parsed.netloc.split(':')[0]
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            # Not an IP address, that's fine (it's a domain)
            pass

        return True
    except Exception:
        return False


def get_dynamic_subvideo_length(width, height):
    """
    Calculate optimal subvideo length based on video resolution for memory efficiency.
    Higher resolution videos need smaller chunks to fit in GPU memory.
    """
    resolution = width * height
    
    if resolution <= 640 * 480:        # 640p and below
        return 100, 12  # subvideo_length, short_clip_len
    elif resolution <= 1280 * 720:     # 720p
        return 80, 8
    elif resolution <= 1920 * 1080:    # 1080p  
        return 60, 6
    elif resolution <= 2560 * 1440:    # 1440p
        return 40, 4
    else:                              # 4K and above
        return 20, 2

def clear_gpu_memory():
    """
    Strategic GPU memory clearing for optimal memory management.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"🧹 GPU memory cleared: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")
    except Exception as e:
        print(f"⚠️  GPU memory clear failed: {e}")
        pass

def performance_checkpoint(stage_name, start_time=None):
    """
    Performance checkpoint logging for identifying bottlenecks.
    """
    import time
    current_time = time.perf_counter()
    
    if start_time is not None:
        elapsed = current_time - start_time
        print(f"⏱️  {stage_name} completed in {elapsed:.2f} seconds")
    else:
        print(f"🚀 {stage_name} started")
    
    return current_time


def _init_gpu_worker(gpu_id):
    """
    Initialize worker process with specific GPU assignment.
    Each worker gets exclusive access to one GPU.
    """
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Worker initialized with GPU {gpu_id}")


def _process_propainter_segment(seg_idx, total_segments, segment, context):
    """
    Process a single smart-cropped segment with faster-propainter.
    Returns frames directory ready for encoding.
    Runs in separate process with assigned GPU via CUDA_VISIBLE_DEVICES.

    Returns a tuple (seg_idx, seg_cleaned_dir, crop_info, temp_dirs_to_cleanup).
    """
    from crop_utils import calculate_crop_region, estimate_speedup

    start_frame, end_frame, seg_bbox = segment
    seg_duration = end_frame - start_frame + 1

    base_name = context['base_name']
    unique_suffix = context['unique_suffix']
    width = context['width']
    height = context['height']
    mask_dir = context['mask_dir']
    original_frames_dir = context['original_frames_dir']

    seg_label = f"{seg_idx + 1}/{total_segments}"
    print(f"\n🎬 Processing segment {seg_label}: frames {start_frame}-{end_frame} ({seg_duration} frames)")

    crop_x, crop_y, crop_w, crop_h = calculate_crop_region(seg_bbox, width, height, padding_ratio=0.2, min_size=128)
    speedup = estimate_speedup((width, height), (crop_w, crop_h))
    print(f"   📐 Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h} (estimated {speedup:.1f}x speedup)")

    seg_prefix = f"{base_name}_{unique_suffix}_seg{seg_idx}"
    seg_frames_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_frames")
    seg_cropped_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cropped")
    seg_mask_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_masks")
    seg_output_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_output")
    seg_cleaned_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cleaned")

    for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
        os.makedirs(path, exist_ok=True)

    try:
        frames_copied = 0
        for frame_idx in range(start_frame, end_frame + 1):
            src = os.path.join(original_frames_dir, f"{frame_idx:04d}.png")
            dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
            if not os.path.exists(src):
                print(f"⚠️  Warning: Frame {frame_idx} ({frame_idx:04d}.png) not found, skipping")
                continue
            shutil.copy2(src, dst)
            frames_copied += 1

        if frames_copied == 0:
            raise RuntimeError(f"No frames copied for segment {seg_idx}")

        print(f"   ✂️  Cropping {frames_copied} frames to {crop_w}x{crop_h}...")
        for frame_idx in range(frames_copied):
            frame_file = f"{frame_idx:04d}.png"
            frame_path = os.path.join(seg_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"⚠️  Warning: Could not read frame {frame_file}, skipping crop")
                continue
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            cv2.imwrite(os.path.join(seg_cropped_dir, frame_file), cropped)

        masks_copied = 0
        for frame_idx in range(start_frame, end_frame + 1):
            mask_file = f"{frame_idx:04d}.png"
            mask_src = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_src):
                mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    cv2.imwrite(os.path.join(seg_mask_dir, f"{masks_copied:04d}.png"), cropped_mask)
                    masks_copied += 1

        print(f"   🎨 Running faster-propainter pipeline on cropped segment...")
        try:
            faster_propainter_path = os.path.join(SCRIPT_DIR, 'faster-propainter-main')
            if faster_propainter_path not in sys.path:
                sys.path.insert(0, faster_propainter_path)
            from watermark import pipeline as faster_propainter_pipeline

            import torch
            use_fp16 = torch.cuda.is_available()

            print(f"   🚀 Direct pipeline: segment {seg_idx+1}, resolution={crop_w}x{crop_h}, neighbor_length=20, ref_stride=10, subvideo_length=80, FP16={use_fp16}")

            # Each process gets its own CUDA context for true parallel processing
            faster_propainter_pipeline(
                video=seg_cropped_dir,
                mask=seg_mask_dir,
                output=seg_output_dir,
                resize_ratio=1.0,
                mask_dilation=4,
                ref_stride=10,
                neighbor_length=20,
                subvideo_length=80,
                raft_iter=20,
                mode="video_inpainting",
                save_frames=True,
                fp16=use_fp16
            )

            print(f"   ✅ faster-propainter segment {seg_idx+1} completed")

        except Exception as exc:
            print(f"❌ faster-propainter failed on segment {seg_idx}: {exc}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"faster-propainter failed on segment {seg_idx}: {exc}") from exc

        seg_propainter_frames = os.path.join(seg_output_dir, os.path.basename(seg_cropped_dir), 'frames')
        if not os.path.exists(seg_propainter_frames):
            raise RuntimeError(f"ProPainter output frames not found for segment {seg_idx}")

        print(f"   🔗 Merging cleaned region back to full frames (in-memory)...")
        # Load all frames into memory first (faster than disk I/O in loop)
        original_frames = []
        cleaned_frames = []
        for frame_idx in range(seg_duration):
            frame_file = f"{frame_idx:04d}.png"
            orig = cv2.imread(os.path.join(seg_frames_dir, frame_file))
            clean = cv2.imread(os.path.join(seg_propainter_frames, frame_file))
            original_frames.append(orig)
            cleaned_frames.append(clean)

        # Merge in memory and write in one pass
        for frame_idx, (original, cleaned_crop) in enumerate(zip(original_frames, cleaned_frames)):
            frame_file = f"{frame_idx:04d}.png"
            if cleaned_crop is not None and original is not None:
                result_frame = original.copy()
                result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop
                cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), result_frame)
            elif original is not None:
                cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), original)

        # Return cleaned frames directory for later encoding
        temp_dirs = [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir]
        return seg_idx, seg_cleaned_dir, {'fps': context['fps']}, temp_dirs

    except Exception as e:
        # Cleanup on error
        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
            shutil.rmtree(path, ignore_errors=True)
        raise


def _encode_segment(seg_idx, total_segments, seg_cleaned_dir, context, temp_dirs_to_cleanup):
    """
    Encode a segment's cleaned frames to video (CPU-bound, runs in parallel).

    Returns a tuple (seg_idx, seg_video_path).
    """
    import subprocess

    base_name = context['base_name']
    unique_suffix = context['unique_suffix']
    fps = context['fps']

    seg_prefix = f"{base_name}_{unique_suffix}_seg{seg_idx}"
    seg_video_path = os.path.join(TEMP_DIR, f"{seg_prefix}.mp4")

    seg_label = f"{seg_idx + 1}/{total_segments}"
    print(f"   🎞️  Encoding segment {seg_label} to video...")

    try:
        encode_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(seg_cleaned_dir, '%04d.png'),
            '-c:v', 'h264_nvenc', '-preset', 'p4', '-b:v', '5M',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            seg_video_path
        ]
        subprocess.run(encode_cmd, capture_output=True, check=True)

        print(f"   ✅ Segment {seg_label} encoded successfully")
        return seg_idx, seg_video_path

    finally:
        # Cleanup temp directories after encoding
        for path in temp_dirs_to_cleanup + [seg_cleaned_dir]:
            shutil.rmtree(path, ignore_errors=True)


def save_detection_debug(image, mask, detections, prefix):
    """
    Save a debug visualization showing YOLO detections and the inpainting mask.

    Args:
        image: Original frame (H, W, 3)
        mask:  Binary mask aligned with image (H, W)
        detections: List of detection dicts with 'bbox'
        prefix: Filename prefix (str)
    """
    if not detections or image is None or mask is None:
        return None

    try:
        overlay = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if mask.shape[:2] != image.shape[:2]:
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        else:
            mask_resized = mask

        mask_color = np.zeros_like(overlay)
        mask_color[:, :, 2] = np.clip(mask_resized, 0, 255)

        debug_image = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)
        filename = f"{prefix}.png"
        output_path = os.path.join(DEBUG_DIR, filename)
        cv2.imwrite(output_path, debug_image)
        print(f"🧪 Detection debug saved: {output_path}")
        return output_path
    except Exception as exc:
        print(f"⚠️  Failed to save detection debug image: {exc}")
        return None

# Initialize Celery
celery = Celery(app.name, broker=app.config['broker_url'], backend=app.config['result_backend'])
celery.conf.update(app.config)

# Celery configuration for production
celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minute timeout
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks (prevent memory leaks)
    result_expires=3600,  # Results expire after 1 hour
    broker_connection_retry_on_startup=True,
    # Fix connection hanging and task pickup blocking
    broker_pool_limit=10,  # Celery default - stable connection pool (was 1 = too restrictive, None = connection churn)
    broker_connection_timeout=10,  # 10 second timeout for broker connection (increased from 3)
    broker_transport_options={
        'visibility_timeout': 300,  # 5 minutes (default 3600) - tasks become visible again after 5min if worker crashes
    },
    result_backend_transport_options={'socket_connect_timeout': 10},
    task_ignore_result=False,  # We need results for status tracking
    task_acks_late=True,
    worker_disable_rate_limits=True,  # Disable rate limiting to prevent task pickup delays
)

# Global model instances (lazy loaded)
detector = None
propainter_ready = False

# Use multiprocessing for true parallel GPU processing
# Each process gets its own CUDA context and model instances
from concurrent.futures import ProcessPoolExecutor

# File cleanup - delete files older than 1 hour
def cleanup_old_files():
    """Delete uploaded and processed files older than 1 hour"""
    import time
    current_time = time.time()
    max_age = 3600  # 1 hour

    for directory in [UPLOAD_DIR, RESULT_DIR]:
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age:
                        os.remove(file_path)
                        print(f"🗑️  Cleaned up old file: {filename} (age: {file_age/60:.1f} min)")
        except Exception as e:
            print(f"⚠️  Cleanup error in {directory}: {e}")

# Schedule cleanup to run every 10 minutes
import threading
def schedule_cleanup():
    cleanup_old_files()
    threading.Timer(600, schedule_cleanup).start()  # Run every 10 minutes

# Start cleanup scheduler
threading.Thread(target=schedule_cleanup, daemon=True).start()
print("🗑️  File cleanup scheduler started (runs every 10 minutes)")

# ============================================================================
# Model Loading (Shared across workers)
# ============================================================================

def _check_propainter_assets() -> bool:
    """
    Verify required ProPainter assets are present.
    Returns True when everything looks good.
    """
    global propainter_ready

    if propainter_ready:
        return True

    required_paths = [
        PROPAINTER_SCRIPT,
        os.path.join(SCRIPT_DIR, 'weights', 'ProPainter.pth'),
        os.path.join(SCRIPT_DIR, 'weights', 'raft-things.pth'),
        os.path.join(SCRIPT_DIR, 'weights', 'recurrent_flow_completion.pth'),
    ]

    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        print("❌ ProPainter assets missing:")
        for path in missing:
            print(f"   - {path}")
        print("   Download weights or copy them into the paths above.")
        propainter_ready = False
        return False

    propainter_ready = True
    return True


def get_detector():
    """
    Lazy load the YOLO detector (TensorRT if available).
    Note: This is used by Celery workers on cloud machines, not the local Flask server.
    """
    global detector

    if detector is None:
        print("=" * 60)
        print("Loading YOLO detector...")
        print("=" * 60)
        from yolo_detector import YOLOWatermarkDetector
        detector = YOLOWatermarkDetector()
        print("=" * 60)
        print("✅ YOLO detector ready!")
        print("=" * 60)

    # Check ProPainter assets (only matters on cloud workers)
    _check_propainter_assets()
    return detector


# ============================================================================
# Celery Tasks (Background Processing)
# ============================================================================

# ============================================================================
# DISTRIBUTED VIDEO PROCESSING TASKS
# These tasks enable multiple workers across different machines to collaborate
# on processing one video together by distributing segments across all GPUs
# ============================================================================

@celery.task(bind=True, name='watermark.prepare_video')
def prepare_video_task(self, video_path, api_base=None, temp_base=None):
    """
    Phase 1: Prepare video for distributed processing
    - Download video if needed
    - Run YOLO detection on all frames (centralized)
    - Generate masks
    - Detect segments (handles moving watermarks)
    - Workers will use these masks or regenerate if not available

    Returns: dict with video_id, segments, metadata for distributed processing
    """
    try:
        import json

        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Preparing video'})

        detector = get_detector()
        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing")

        # Download video if on remote worker
        if not os.path.exists(video_path):
            # Prefer dynamic base provided by API, else env
            tunnel = api_base or os.getenv('TUNNEL_URL')
            if tunnel:
                try:
                    from urllib.parse import urljoin
                    import requests
                    from pathlib import PureWindowsPath
                    base_name = PureWindowsPath(video_path).name if '\\' in video_path else os.path.basename(video_path)
                    download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                    print(f"🌐 Downloading video: {download_url}")
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                    r.raise_for_status()
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    video_path = os.path.join(UPLOAD_DIR, base_name)
                    with open(video_path, 'wb') as f:
                        f.write(r.content)
                    print(f"✅ Video downloaded: {video_path}")
                except Exception as e:
                    raise Exception(f"Failed to download video: {e}")
            else:
                raise Exception(f"Video not found: {video_path}")

        print(f"📹 Preparing video: {video_path} ({os.path.getsize(video_path) / (1024 * 1024):.2f} MB)")

        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Analyzing video'})

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        base_name = Path(video_path).stem
        video_id = self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8]

        # Create shared directories for distributed access
        shared_mask_dir = os.path.join(PROPAINTER_MASK_ROOT, f"{base_name}_{video_id}")
        shared_frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_originals")
        os.makedirs(shared_mask_dir, exist_ok=True)
        os.makedirs(shared_frames_dir, exist_ok=True)

        print(f"🎯 Running YOLO detection on {total_frames} frames...")
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Detecting watermarks'})

        zero_mask = np.zeros((height, width), dtype=np.uint8)
        last_valid_bbox = None
        frames_processed = 0
        frames_with_watermark = 0
        detect_interval = 3
        warmup_frames = 60
        hit_found = False
        det_conf = 0.15
        bboxes_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            progress = 10 + int((frames_processed / max(total_frames, 1)) * 30)
            self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'Detection {frames_processed}/{total_frames}'})

            if (frames_processed % detect_interval == 0) or (not hit_found and frames_processed < warmup_frames):
                detections = detector.detect(frame, confidence_threshold=det_conf, padding=0)
                if detections:
                    frames_with_watermark += 1
                    last_valid_bbox = detections[0]['bbox']
                    hit_found = True
                    bboxes_per_frame.append(last_valid_bbox)
                elif last_valid_bbox:
                    bboxes_per_frame.append(last_valid_bbox)
                else:
                    bboxes_per_frame.append(None)

                mask = detector.create_mask(frame, detections) if detections else zero_mask
            else:
                bboxes_per_frame.append(last_valid_bbox)
                mask = detector.create_mask(frame, [{'bbox': last_valid_bbox}]) if last_valid_bbox else zero_mask

            mask_path = os.path.join(shared_mask_dir, f"{frames_processed:04d}.png")
            cv2.imwrite(mask_path, mask)
            frames_processed += 1

        cap.release()

        if frames_processed == 0:
            raise RuntimeError("No frames processed - video may be corrupted")

        print(f"✅ Detection complete: {frames_processed} frames, {frames_with_watermark} with watermarks")

        # Detect segments (by watermark position changes)
        from segment_detector import detect_segments, merge_adjacent_segments
        segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
        if segments:
            segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
            print(f"📊 Detected {len(segments)} segments for distributed processing")
            for i, (start, end, bbox) in enumerate(segments):
                print(f"   Segment {i+1}: frames {start}-{end} ({end-start+1} frames), bbox={bbox}")
        else:
            # No segments detected - treat entire video as one segment
            segments = [(0, frames_processed-1, last_valid_bbox if last_valid_bbox else [0,0,width,height])]
            print("📊 No segments detected - processing entire video as one segment")

        # Optional: force time-based splitting to ensure multi-GPU distribution
        try:
            import math
            min_segments = int(os.getenv('MIN_SEGMENTS', '0'))
            min_chunk_frames = int(os.getenv('MIN_CHUNK_FRAMES', '60'))
        except Exception:
            min_segments = 0
            min_chunk_frames = 60

        if min_segments and len(segments) < min_segments and frames_processed >= min_chunk_frames:
            # Split the longest segment or the only segment into at least min_segments time chunks
            base_seg = max(segments, key=lambda s: (s[1]-s[0]+1)) if segments else (0, frames_processed-1, last_valid_bbox if last_valid_bbox else [0,0,width,height])
            s0, e0, bb = base_seg
            duration = e0 - s0 + 1
            num_chunks = min_segments
            chunk = max(min_chunk_frames, math.ceil(duration / num_chunks))
            new_segments = []
            cur = s0
            while cur <= e0:
                end = min(e0, cur + chunk - 1)
                new_segments.append((cur, end, bb if bb else [0,0,width,height]))
                cur = end + 1
                if len(new_segments) >= num_chunks and end < e0:
                    # If more frames remain after hitting desired count, extend last chunk
                    new_segments[-1] = (new_segments[-1][0], e0, new_segments[-1][2])
                    break
            segments = new_segments
            print(f"🪓 Force-split enabled: created {len(segments)} time chunks (chunk≈{chunk} frames)")

        # Provide a base URL so OTHER workers can fetch frames/masks from this host
        temp_base_url = temp_base or os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL')
        if temp_base_url:
            print(f"🌐 Shared temp base set for workers: {temp_base_url.rstrip('/')}/temp/")
        else:
            print("⚠️  No TEMP_BASE_URL/TUNNEL_URL set; only preparing worker can read local frames.")

        # Prepare segment data for distribution
        segment_tasks_data = []
        api_base_url = api_base or os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL')
        for seg_idx, (start_frame, end_frame, bbox) in enumerate(segments):
            segment_tasks_data.append({
                'seg_idx': seg_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'bbox': bbox,
                'video_id': video_id,
                'base_name': base_name,
                'width': width,
                'height': height,
                'fps': fps,
                'shared_mask_dir': shared_mask_dir,
                'video_url': f"{api_base_url.rstrip('/')}/uploads/{os.path.basename(video_path)}",
                'temp_base_url': temp_base_url,
                'api_base': api_base_url,
                'upload_filename': os.path.basename(video_path),
            })

        result = {
            'video_id': video_id,
            'base_name': base_name,
            'video_path': video_path,
            'segments': segment_tasks_data,
            'total_segments': len(segments),
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': frames_processed,
            'frames_with_watermark': frames_with_watermark,
            'shared_mask_dir': shared_mask_dir,
            'shared_frames_dir': shared_frames_dir,
            'api_base': api_base or os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL'),
            'temp_base_url': temp_base_url,
        }

        print(f"✅ Video prepared for distributed processing: {len(segments)} segments ready")

        # Manually dispatch each segment task to guarantee they all get queued
        # This is more reliable than chord with solo pool
        print(f"🔥 Dispatching {len(segments)} segment tasks manually across all workers...")
        self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Dispatching {len(segments)} parallel tasks'})

        # Add total_segments to each segment data
        for seg in segment_tasks_data:
            seg['total_segments'] = len(segments)

        # Create a tracking key in Redis for this video
        tracking_key = f"segments:{video_id}"
        celery.backend.set(tracking_key, 0)  # Initialize counter to 0
        celery.backend.set(f"{tracking_key}:total", len(segments))  # Store total
        celery.backend.set(f"{tracking_key}:prepare_result", json.dumps(result))  # Store prepare result

        # Dispatch each segment task individually - guarantees queue delivery
        segment_tasks = []
        for seg_data in segment_tasks_data:
            # Add tracking info to segment data
            seg_data['tracking_key'] = tracking_key

            task = process_segment_task.apply_async(args=[seg_data])
            segment_tasks.append(task.id)

            # Store task ID in Redis for result collection later (1 hour TTL)
            celery.backend.client.setex(f"{tracking_key}:task_{seg_data['seg_idx']}", 3600, task.id)

            print(f"   ✅ Segment {seg_data['seg_idx']+1} task queued: {task.id}")

        print(f"✅ All {len(segments)} segment tasks dispatched to queue!")
        print(f"   Workers will grab and process them in parallel")
        print(f"   Tracking progress with key: {tracking_key}")

        # Create a fake task ID for frontend tracking (we'll track the segments)
        tracking_task_id = f"distributed_{video_id}"

        # Return tracking ID for frontend
        return {
            'chord_id': tracking_task_id,
            'status': 'processing',
            'message': f'Distributed processing started: {len(segments)} segments across all workers'
        }

    except Exception as e:
        print(f"❌ Error preparing video: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.process_segment')
def process_segment_task(self, segment_data):
    """
    Phase 2: Process one segment (distributed across all available workers/GPUs)
    - Download required frames and masks
    - Run ProPainter on this segment
    - Upload cleaned frames back

    This task can run on ANY available worker with a GPU
    """
    try:
        seg_idx = segment_data['seg_idx']
        total_segments = segment_data.get('total_segments', '?')
        start_frame = segment_data['start_frame']
        end_frame = segment_data['end_frame']
        bbox = segment_data['bbox']
        video_id = segment_data['video_id']
        base_name = segment_data['base_name']
        width = segment_data['width']
        height = segment_data['height']
        fps = segment_data['fps']

        print(f"\n🎬 Worker processing segment {seg_idx+1}/{total_segments}: frames {start_frame}-{end_frame}")
        self.update_state(state='STARTED', meta={'progress': 0, 'status': f'Processing segment {seg_idx+1}'})

        # Import required modules
        import subprocess
        from crop_utils import calculate_crop_region

        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"   📐 Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

        seg_duration = end_frame - start_frame + 1

        # Create local temp directories for this segment
        seg_prefix = f"{base_name}_{video_id}_seg{seg_idx}"
        seg_frames_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_frames")
        seg_cropped_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cropped")
        seg_mask_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_masks")
        seg_output_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_output")
        seg_cleaned_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cleaned")

        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
            os.makedirs(path, exist_ok=True)

        # Each worker extracts ONLY its assigned frames directly from video (parallel speedup!)
        origin_base = segment_data.get('temp_base_url') or os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL')
        shared_mask_dir = segment_data.get('shared_mask_dir')

        print(f"   📥 Extracting frames {start_frame}-{end_frame} from video...")
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Extracting assigned frames'})

        video_url = segment_data.get('video_url')
        if not video_url:
            raise RuntimeError("No video_url provided in segment_data")

        # Download video
        print(f"   ⬇️  Downloading video: {video_url}")
        import requests
        r = requests.get(video_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
        r.raise_for_status()

        upload_filename = segment_data.get('upload_filename', 'temp_video.mp4')
        local_video = os.path.join(TEMP_DIR, f"seg_{seg_idx}_{upload_filename}")
        with open(local_video, 'wb') as f:
            f.write(r.content)

        # Extract ONLY frames for this segment (start_frame to end_frame)
        print(f"   🎬 Extracting frames {start_frame}-{end_frame}...")
        cap = cv2.VideoCapture(local_video)
        if not cap.isOpened():
            raise RuntimeError("Failed to open downloaded video")

        frames_copied = 0
        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame > end_frame:  # Stop after segment end
                break
            if current_frame >= start_frame:  # Only save frames in segment range
                dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
                cv2.imwrite(dst, frame)
                frames_copied += 1
            current_frame += 1

        cap.release()
        os.remove(local_video)  # Clean up downloaded video

        if frames_copied == 0:
            raise RuntimeError(f"No frames extracted for segment {seg_idx}")

        print(f"   ✅ Extracted {frames_copied} frames ({start_frame}-{end_frame})")

        # Try to download masks from shared location (same PC) or regenerate them (different PC)
        masks_downloaded = False
        if origin_base and shared_mask_dir:
            print(f"   📥 Attempting to download masks from shared location...")
            self.update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Downloading masks'})

            try:
                import requests
                masks_needed = list(range(start_frame, end_frame + 1))
                masks_succeeded = 0

                for abs_frame_idx in masks_needed:
                    mask_filename = f"{abs_frame_idx:04d}.png"
                    mask_url = f"{origin_base.rstrip('/')}/temp/{os.path.basename(shared_mask_dir)}/{mask_filename}"

                    try:
                        r = requests.get(mask_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=10)
                        r.raise_for_status()

                        # Save to local segment mask dir with LOCAL index (0000, 0001, etc.)
                        local_idx = abs_frame_idx - start_frame
                        local_mask_path = os.path.join(seg_mask_dir, f"{local_idx:04d}.png")
                        with open(local_mask_path, 'wb') as f:
                            f.write(r.content)
                        masks_succeeded += 1
                    except Exception as mask_err:
                        # Silently skip - we'll regenerate if needed
                        break

                if masks_succeeded == len(masks_needed):
                    print(f"   ✅ Downloaded {masks_succeeded} masks from shared location")
                    masks_downloaded = True
                else:
                    print(f"   ⚠️  Only {masks_succeeded}/{len(masks_needed)} masks downloaded - will regenerate")
            except Exception as download_err:
                print(f"   ⚠️  Mask download failed: {download_err} - will regenerate")

        # Fallback: Regenerate masks if not downloaded
        detector = None
        if not masks_downloaded:
            print(f"   🎯 Regenerating masks with YOLO detection on {frames_copied} frames...")
            self.update_state(state='PROCESSING', meta={'progress': 25, 'status': f'Detecting watermarks'})

            detector = get_detector()
            last_valid_bbox = None
            frames_with_watermark = 0
            detect_interval = 3
            warmup_frames = min(60, frames_copied)
            hit_found = False
            det_conf = 0.15

            for frame_idx in range(frames_copied):
                frame_file = f"{frame_idx:04d}.png"
                frame_path = os.path.join(seg_frames_dir, frame_file)
                frame = cv2.imread(frame_path)

                if frame is None:
                    continue

                # Detect watermark (don't create masks yet - will do after cropping)
                if (frame_idx % detect_interval == 0) or (not hit_found and frame_idx < warmup_frames):
                    detections = detector.detect(frame, confidence_threshold=det_conf, padding=0)
                    if detections:
                        frames_with_watermark += 1
                        last_valid_bbox = detections[0]['bbox']
                        hit_found = True

            print(f"   ✅ Detection complete: {frames_with_watermark}/{frames_copied} frames with watermarks")
        else:
            # Masks were downloaded - need to extract bbox from them
            print(f"   📊 Using downloaded masks - extracting bbox info...")
            last_valid_bbox = bbox  # Use bbox from segment_data (from centralized detection)

        # Update crop region based on detected bbox (calculate_crop_region handles min sizing)
        if last_valid_bbox:
            # Recalculate crop region with detected bbox (includes min_size=128 expansion)
            crop_x, crop_y, crop_w, crop_h = calculate_crop_region(last_valid_bbox, width, height, padding_ratio=0.2, min_size=128)
            print(f"   📐 Detected crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        else:
            print(f"   ℹ️  No watermark detected in this chunk - will skip ProPainter")

        # Crop frames to detected watermark region AND create masks on cropped frames
        if last_valid_bbox:
            print(f"   ✂️  Cropping frames to watermark region...")
            for frame_idx in range(frames_copied):
                frame_file = f"{frame_idx:04d}.png"
                frame_path = os.path.join(seg_frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    cv2.imwrite(os.path.join(seg_cropped_dir, frame_file), cropped)

            # Create masks on CROPPED frames (not full frames!)
            print(f"   🎭 Creating masks on cropped frames...")
            if not masks_downloaded and detector:
                # Calculate bbox relative to cropped region
                x1, y1, x2, y2 = last_valid_bbox
                # Translate bbox from full frame coords to crop coords
                crop_bbox_x1 = max(0, x1 - crop_x)
                crop_bbox_y1 = max(0, y1 - crop_y)
                crop_bbox_x2 = min(crop_w, x2 - crop_x)
                crop_bbox_y2 = min(crop_h, y2 - crop_y)
                crop_bbox = [crop_bbox_x1, crop_bbox_y1, crop_bbox_x2, crop_bbox_y2]

                # Create masks on cropped frames
                for frame_idx in range(frames_copied):
                    frame_file = f"{frame_idx:04d}.png"
                    cropped_frame_path = os.path.join(seg_cropped_dir, frame_file)
                    cropped_frame = cv2.imread(cropped_frame_path)
                    if cropped_frame is not None:
                        # Create mask on cropped frame with relative bbox
                        mask = detector.create_mask(cropped_frame, [{'bbox': crop_bbox}])
                        mask_path = os.path.join(seg_mask_dir, frame_file)
                        cv2.imwrite(mask_path, mask)
            else:
                # Masks were downloaded - crop them to the region
                print(f"   ✂️  Cropping downloaded masks to region...")
                for frame_idx in range(frames_copied):
                    mask_file = f"{frame_idx:04d}.png"
                    mask_path = os.path.join(seg_mask_dir, mask_file)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        cv2.imwrite(mask_path, cropped_mask)

        print(f"   ✅ Prepared {frames_copied} frames and masks")

        # VALIDATION: Check first mask to catch bugs early
        if last_valid_bbox and frames_with_watermark > 0:
            first_mask_path = os.path.join(seg_mask_dir, "0000.png")
            if os.path.exists(first_mask_path):
                test_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
                if test_mask is not None:
                    white_pixels = np.sum(test_mask == 255)
                    total_pixels = test_mask.size
                    white_pct = (white_pixels / total_pixels) * 100

                    print(f"   📊 Mask validation: {white_pct:.1f}% white pixels (target region)")

                    # Warn if mask seems wrong
                    if white_pct > 50:
                        print(f"   ⚠️  WARNING: Mask is {white_pct:.1f}% white - suspicious! May cause black output.")
                    elif white_pct < 0.1:
                        print(f"   ⚠️  WARNING: Mask is {white_pct:.1f}% white - almost empty! No inpainting will occur.")

        # CONDITIONAL: Only run ProPainter if watermark was detected
        if not last_valid_bbox or frames_with_watermark == 0:
            print(f"   ⏭️  No watermark detected - skipping ProPainter, encoding original frames")
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'No watermark - encoding original'})

            # Just copy original frames to cleaned dir (no processing needed)
            for frame_idx in range(frames_copied):
                frame_file = f"{frame_idx:04d}.png"
                src = os.path.join(seg_frames_dir, frame_file)
                dst = os.path.join(seg_cleaned_dir, frame_file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        else:
            # Run ProPainter on this segment - watermark detected!
            print(f"   🎨 Running ProPainter on {frames_with_watermark} watermarked frames...")
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Running ProPainter'})

            try:
                faster_propainter_path = os.path.join(SCRIPT_DIR, 'faster-propainter-main')
                if faster_propainter_path not in sys.path:
                    sys.path.insert(0, faster_propainter_path)

                print(f"   📦 Importing ProPainter from: {faster_propainter_path}")
                from watermark import pipeline as faster_propainter_pipeline
                print(f"   ✅ ProPainter import successful")

                import torch
                use_fp16 = torch.cuda.is_available()
                print(f"   🔧 GPU available: {use_fp16}, Running ProPainter pipeline...")

                faster_propainter_pipeline(
                    video=seg_cropped_dir,
                    mask=seg_mask_dir,
                    output=seg_output_dir,
                    resize_ratio=1.0,
                    mask_dilation=4,
                    ref_stride=10,
                    neighbor_length=20,
                    subvideo_length=80,
                    raft_iter=20,
                    mode="video_inpainting",
                    save_frames=True,
                    fp16=use_fp16
                )

                print(f"   ✅ ProPainter complete for segment {seg_idx+1}")
            except Exception as e:
                print(f"❌ ProPainter failed on segment {seg_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                clear_gpu_memory()

            # Merge cleaned region back to full frames
            print(f"   🔗 Merging cleaned region (in-memory)...")
            self.update_state(state='PROCESSING', meta={'progress': 80, 'status': f'Merging results'})

            seg_propainter_frames = os.path.join(seg_output_dir, os.path.basename(seg_cropped_dir), 'frames')
            if not os.path.exists(seg_propainter_frames):
                raise RuntimeError(f"ProPainter output not found for segment {seg_idx}")

            # Load all frames into memory first (faster than disk I/O in loop)
            original_frames = []
            cleaned_frames = []
            for frame_idx in range(seg_duration):
                frame_file = f"{frame_idx:04d}.png"
                orig = cv2.imread(os.path.join(seg_frames_dir, frame_file))
                clean = cv2.imread(os.path.join(seg_propainter_frames, frame_file))
                original_frames.append(orig)
                cleaned_frames.append(clean)

            # Merge in memory and write in one pass
            for frame_idx, (original, cleaned_crop) in enumerate(zip(original_frames, cleaned_frames)):
                frame_file = f"{frame_idx:04d}.png"
                if cleaned_crop is not None and original is not None:
                    result_frame = original.copy()
                    result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop
                    cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), result_frame)
                elif original is not None:
                    cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), original)

        # Encode segment to video
        print(f"   🎞️  Encoding segment...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': f'Encoding video'})

        seg_video_path = os.path.join(RESULT_DIR, f"{seg_prefix}.mp4")
        encode_cmd = [
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(seg_cleaned_dir, '%04d.png'),
            '-c:v', 'h264_nvenc', '-preset', 'p4', '-b:v', '5M',
            '-pix_fmt', 'yuv420p', '-profile:v', 'main',
            seg_video_path
        ]
        subprocess.run(encode_cmd, capture_output=True, check=True)

        # Skip upload if on same machine (all workers on 1 GPU = same machine!)
        api_base = segment_data.get('api_base') or os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL') or origin_base
        # If no external API base or it's ngrok (same machine), skip upload
        is_local = api_base is None or 'localhost' in (api_base or '').lower() or '127.0.0.1' in (api_base or '') or 'ngrok' in (api_base or '').lower()

        if not is_local and api_base:
            try:
                upload_url = f"{api_base.rstrip('/')}/api/upload-segment"
                print(f"   ⬆️  Uploading segment {seg_idx+1} to API server...")
                with open(seg_video_path, 'rb') as f:
                    resp = requests.post(
                        upload_url,
                        headers={'ngrok-skip-browser-warning': 'true'},
                        files={'file': (os.path.basename(seg_video_path), f, 'video/mp4')},
                        data={'video_id': video_id, 'seg_idx': seg_idx},
                        timeout=120
                    )
                if resp.ok:
                    print(f"   ✅ Segment {seg_idx+1} uploaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to upload segment: {e}")
        else:
            print(f"   ✅ Segment {seg_idx+1} stored locally (skip upload)")

        # Cleanup temp directories
        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
            shutil.rmtree(path, ignore_errors=True)

        result = {
            'seg_idx': seg_idx,
            'video_path': seg_video_path,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frames_processed': seg_duration,
        }

        print(f"✅ Segment {seg_idx+1}/{total_segments} complete!")

        # Note: Don't call self.update_state(state='SUCCESS') - it overrides the return value!
        # Celery automatically sets state to SUCCESS when the task returns normally

        # Track completion in Redis for manual coordination
        tracking_key = segment_data.get('tracking_key')
        if tracking_key:
            import json

            # FIRST: Store this segment's result in Redis (before incrementing counter)
            # This prevents race condition where counter = total but results aren't ready yet
            celery.backend.client.setex(f"{tracking_key}:result_{seg_idx}", 3600, json.dumps(result))

            # THEN: Increment counter atomically
            completed = celery.backend.client.incr(tracking_key)
            total = int(celery.backend.get(f"{tracking_key}:total") or 0)

            print(f"📊 Tracking: {completed}/{total} segments complete")

            # Check if this is the last segment
            if completed >= total:
                # Use atomic lock to ensure only ONE worker triggers finalize
                lock_key = f"{tracking_key}:finalize_lock"
                if celery.backend.client.setnx(lock_key, 1):
                    # This worker won the race - trigger finalize
                    celery.backend.client.expire(lock_key, 60)  # Lock expires in 60s

                    print(f"🎉 All segments complete! This worker triggering finalize...")

                    # Collect all segment results from Redis (stored before counter increment)
                    # This avoids race conditions with AsyncResult
                    segment_results = []
                    for i in range(total):
                        result_json = celery.backend.get(f"{tracking_key}:result_{i}")
                        if result_json:
                            result_data = json.loads(result_json)
                            segment_results.append(result_data)
                            print(f"   ✅ Collected segment {i} result: {result_data['video_path']}")
                        else:
                            print(f"⚠️  Segment {i} result not found in Redis")

                    # Get prepare result
                    prepare_result_json = celery.backend.get(f"{tracking_key}:prepare_result")
                    if prepare_result_json:
                        prepare_result = json.loads(prepare_result_json)

                        if len(segment_results) == total:
                            finalize_video_task.apply_async(args=[segment_results, prepare_result])
                            print(f"✅ Finalize task triggered with {len(segment_results)} segment results!")
                        else:
                            print(f"❌ Cannot finalize: expected {total} segments, got {len(segment_results)}")
                else:
                    print(f"⏭️  Another worker already triggered finalize")

        return result

    except Exception as e:
        print(f"❌ Error processing segment {seg_idx}: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.finalize_video')
def finalize_video_task(self, segment_results, prepare_result):
    """
    Phase 3: Finalize video after all segments are processed
    - Collect all segment videos
    - Concatenate in correct order
    - Merge audio from original
    - Upload final result
    """
    try:
        import requests
        import subprocess
        import shutil

        video_id = prepare_result['video_id']
        base_name = prepare_result['base_name']
        video_path = prepare_result['video_path']
        fps = prepare_result['fps']
        total_segments = prepare_result['total_segments']

        print(f"\n🎬 Finalizing video: {base_name} ({total_segments} segments)")
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Finalizing video'})

        # Sort segments by index
        segment_results = sorted(segment_results, key=lambda x: x['seg_idx'])

        # Download segment videos if needed
        tunnel = prepare_result.get('api_base') or os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL')
        segment_videos = []

        print(f"📥 Collecting {total_segments} segment videos...")
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Collecting segments'})

        for seg_result in segment_results:
            seg_video_path = seg_result['video_path']

            # Check if segment exists locally
            if os.path.exists(seg_video_path):
                segment_videos.append(seg_video_path)
            elif tunnel:
                # Download from API server
                try:
                    seg_filename = os.path.basename(seg_video_path)
                    download_url = f"{tunnel.rstrip('/')}/results/{seg_filename}"
                    print(f"   ⬇️  Downloading segment {seg_result['seg_idx']+1}...")

                    import requests
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                    r.raise_for_status()

                    local_path = os.path.join(TEMP_DIR, seg_filename)
                    with open(local_path, 'wb') as f:
                        f.write(r.content)
                    segment_videos.append(local_path)
                except Exception as e:
                    raise Exception(f"Failed to download segment {seg_result['seg_idx']}: {e}")
            else:
                raise Exception(f"Segment video not found: {seg_video_path}")

        if len(segment_videos) != total_segments:
            raise RuntimeError(f"Expected {total_segments} segments, got {len(segment_videos)}")

        print(f"✅ All {total_segments} segments collected")

        # Concatenate segments
        print(f"🔗 Concatenating segments...")
        self.update_state(state='PROCESSING', meta={'progress': 40, 'status': 'Concatenating segments'})

        concat_list_file = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_concat.txt")
        with open(concat_list_file, 'w') as f:
            for seg_video in segment_videos:
                f.write(f"file '{seg_video}'\n")

        temp_processed = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_processed.mp4")

        import subprocess
        concat_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_list_file,
            '-c', 'copy',
            temp_processed
        ]
        subprocess.run(concat_cmd, capture_output=True, check=True)

        print(f"✅ Segments concatenated")

        # Merge audio from original video
        print(f"🎵 Merging audio...")
        self.update_state(state='PROCESSING', meta={'progress': 70, 'status': 'Merging audio'})

        final_output = os.path.join(RESULT_DIR, f"{base_name}_propainter.mp4")

        # Check if original has audio
        check_audio_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
        has_audio = 'audio' in has_audio_check.stdout

        if has_audio:
            merge_cmd = [
                'ffmpeg', '-y',
                '-i', temp_processed,
                '-i', video_path,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',  # Don't re-encode video, just copy!
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                final_output
            ]
        else:
            merge_cmd = [
                'ffmpeg', '-y',
                '-i', temp_processed,
                '-c:v', 'copy',  # No audio, just copy video stream
                final_output
            ]

        audio_result = subprocess.run(merge_cmd, capture_output=True, text=True, timeout=300)
        if audio_result.returncode != 0:
            print("⚠️  Audio merge failed, using video without audio")
            final_output = temp_processed
        else:
            os.remove(temp_processed)

        # Cleanup
        print(f"🧹 Cleaning up...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Cleaning up'})

        for seg_video in segment_videos:
            if os.path.exists(seg_video):
                os.remove(seg_video)
        os.remove(concat_list_file)

        # Cleanup shared directories
        shared_mask_dir = prepare_result.get('shared_mask_dir')
        shared_frames_dir = prepare_result.get('shared_frames_dir')
        if shared_mask_dir:
            shutil.rmtree(shared_mask_dir, ignore_errors=True)
        if shared_frames_dir:
            shutil.rmtree(shared_frames_dir, ignore_errors=True)

        # Upload final result
        uploaded_path = None
        if tunnel and os.getenv('UPLOAD_RESULT_BACK', '1') == '1':
            try:
                upload_url = f"{tunnel.rstrip('/')}/api/upload-result"
                print(f"⬆️  Uploading final result to API server...")
                with open(final_output, 'rb') as f:
                    resp = requests.post(
                        upload_url,
                        headers={'ngrok-skip-browser-warning': 'true'},
                        files={'file': (os.path.basename(final_output), f, 'video/mp4')},
                        timeout=120
                    )
                if resp.ok:
                    j = resp.json()
                    if j.get('status') == 'success' and j.get('result_url'):
                        uploaded_path = j['result_url']
                        print(f"✅ Final result uploaded")
            except Exception as e:
                print(f"⚠️  Upload failed: {e}")

        result = {
            'path': uploaded_path or final_output,
            'metadata': {
                'video_id': video_id,
                'total_segments': total_segments,
                'total_frames': prepare_result['total_frames'],
                'frames_with_watermark': prepare_result['frames_with_watermark'],
                'fps': fps,
                'width': prepare_result['width'],
                'height': prepare_result['height'],
            }
        }

        print(f"✅ Video finalized: {final_output}")
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})
        return result

    except Exception as e:
        print(f"❌ Error finalizing video: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.coordinate_segments')
def coordinate_segments_task(self, segment_task_ids, prepare_result):
    """
    Coordinator task that waits for all segment tasks to complete,
    then triggers finalize_video_task.

    This replaces chord functionality with reliable manual coordination.
    """
    try:
        from celery.result import AsyncResult

        total_segments = len(segment_task_ids)
        print(f"📊 Coordinator: Waiting for {total_segments} segment tasks to complete...")
        self.update_state(state='STARTED', meta={'progress': 0, 'status': f'Waiting for {total_segments} segments'})

        # Wait for all segment tasks to complete
        segment_results = []
        for i, task_id in enumerate(segment_task_ids):
            print(f"   ⏳ Waiting for segment task {i+1}/{total_segments}: {task_id}")
            task_result = AsyncResult(task_id, app=celery)

            # Wait for this segment to complete (blocks, but doesn't block workers from processing)
            result = task_result.get(timeout=3600)  # 1 hour timeout per segment
            segment_results.append(result)

            progress = int((i + 1) / total_segments * 80)
            self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'{i+1}/{total_segments} segments complete'})
            print(f"   ✅ Segment {i+1}/{total_segments} complete!")

        print(f"✅ All {total_segments} segments complete! Launching finalize task...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Finalizing video'})

        # All segments done - now finalize
        final_result = finalize_video_task.apply_async(args=[segment_results, prepare_result])

        # Wait for finalize to complete
        final_output = final_result.get(timeout=600)  # 10 minute timeout

        print(f"✅ Video processing complete!")
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})

        return final_output

    except Exception as e:
        print(f"❌ Coordinator failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark._launch_segments')
def launch_segments_task(self, prepare_result):
    """Create a chord of segment tasks and launch them.
    Returns the chord result for tracking.
    """
    try:
        from celery import chord, group

        if not prepare_result or 'segments' not in prepare_result:
            raise RuntimeError("Video preparation failed")

        segments = prepare_result['segments']
        total_segments = len(segments)

        # Add total_segments to each segment data
        for seg in segments:
            seg['total_segments'] = total_segments

        print(f"🔥 Launching chord: {total_segments} segments in parallel across all workers...")

        # Create the chord
        header = group([process_segment_task.s(seg) for seg in segments])
        callback = finalize_video_task.s(prepare_result)

        # Apply the chord asynchronously - this dispatches tasks immediately
        chord_result = chord(header)(callback)

        print(f"✅ Chord dispatched: {total_segments} segment tasks queued")
        print(f"   Workers will pick up tasks and process in parallel")

        # Wait for the chord to complete and return the final result
        # This blocks this task but doesn't block workers from processing segments
        final_result = chord_result.get(timeout=3600)

        print(f"✅ All segments processed and finalized!")
        return final_result

    except Exception as e:
        print(f"❌ Segment launch failed: {e}")
        import traceback; traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.remove_video_distributed')
def process_video_distributed_task(self, video_path, api_base=None, temp_base=None):
    """
    DISTRIBUTED VIDEO PROCESSING - Main entry point

    This task coordinates multiple workers across different machines to process
    one video together. It uses Celery's chord pattern to:
    1. Prepare video (YOLO detection, segmentation) - runs on one worker
    2. Process segments in parallel - distributes across ALL available workers/GPUs
    3. Finalize video (concatenate, merge audio) - runs on one worker

    All workers (across all computers) will collaborate on the same video job.
    """
    try:
        from celery import chain, chord, group

        print(f"🚀 Starting DISTRIBUTED video processing: {video_path}")
        print(f"   All available workers will collaborate on this video")

        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Initializing distributed processing'})

        # Phase 1: Prepare video (runs on one worker)
        print("📋 Phase 1: Preparing video for distribution...")
        prepare_result = prepare_video_task.apply_async(
            args=[video_path],
            kwargs={'api_base': api_base, 'temp_base': temp_base}
        ).get()

        if not prepare_result or 'segments' not in prepare_result:
            raise RuntimeError("Video preparation failed")

        segments = prepare_result['segments']
        total_segments = len(segments)

        print(f"✅ Video prepared: {total_segments} segments ready for distributed processing")
        print(f"🌐 Distributing segments across all available workers...")

        self.update_state(
            state='PROCESSING',
            meta={'progress': 50, 'status': f'Distributing {total_segments} segments across workers'}
        )

        # Add total_segments to each segment data
        for seg_data in segments:
            seg_data['total_segments'] = total_segments

        # Phase 2: Process segments in parallel (distributes across ALL workers)
        # Using chord: when all segment tasks complete, finalize task runs automatically
        print(f"🔥 Phase 2: Processing {total_segments} segments in parallel...")

        workflow = chord(
            # Header: All segment processing tasks (run in parallel across all workers)
            group([
                process_segment_task.s(seg_data)
                for seg_data in segments
            ]),
            # Callback: Finalization task (runs when all segments complete)
            finalize_video_task.s(prepare_result)
        )

        # Execute the distributed workflow
        result = workflow.apply_async()

        print(f"📊 Workflow submitted:")
        print(f"   - {total_segments} segment tasks queued")
        print(f"   - Tasks will be picked up by ANY available worker")
        print(f"   - Finalization will run automatically when all segments complete")

        # Wait for the entire workflow to complete
        final_result = result.get(timeout=3600)  # 1 hour timeout

        print(f"✅ DISTRIBUTED processing complete!")
        print(f"   Final video: {final_result.get('path')}")

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Distributed processing complete'})

        return final_result

    except Exception as e:
        print(f"❌ Distributed processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# ORIGINAL MONOLITHIC TASKS (kept for backward compatibility)
# ============================================================================

@celery.task(bind=True, name='watermark.remove_image')
def process_image_task(self, image_path):
    """
    Single-image watermark removal using YOLO mask + ProPainter with FP16.

    ProPainter works on video sequences, so we:
    1. Duplicate the image into a 3-frame sequence
    2. Generate masks for each frame
    3. Run ProPainter with --fp16
    4. Extract the middle frame as result
    """
    try:
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Loading detector'})
        det = get_detector()

        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing - see logs for details")

        # If running on a remote worker (e.g., Salad), the local path from the API host
        # won't exist. In that case, try to download the file from the API via TUNNEL_URL.
        if not os.path.exists(image_path):
            tunnel = os.getenv('TUNNEL_URL')
            try:
                from urllib.parse import urljoin
                import requests
            except Exception:
                tunnel = None

            if tunnel:
                # Handle Windows-style paths coming from the API host
                try:
                    from pathlib import PureWindowsPath
                    base_name = PureWindowsPath(image_path).name
                except Exception:
                    base_name = os.path.basename(image_path.replace('\\', '/'))

                download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                print(f"🌐 Image not found locally. Downloading from: {download_url}")
                try:
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=30)
                    r.raise_for_status()
                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    image_path = remote_cached
                    print(f"✅ Downloaded to: {image_path}")
                except Exception as dl_err:
                    print(f"❌ Failed to download image from API host: {dl_err}")
                    raise Exception(f"Image file not found and remote download failed: {base_name}")
            else:
                raise Exception(f"Image file not found: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception('Failed to read image')

        height, width = img.shape[:2]
        base_name = Path(image_path).stem
        unique_suffix = self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8]

        self.update_state(state='PROCESSING', meta={'progress': 15, 'status': 'Detecting watermark'})
        detection_start = performance_checkpoint("YOLO Detection")
        detections = det.detect(img, confidence_threshold=0.20, padding=0)  # Lower threshold for faint watermarks (Sora optimized)
        performance_checkpoint("YOLO Detection", detection_start)

        # If no detections, return original
        if not detections:
            out_name = base_name + '_clean.png'
            out_path = os.path.join(RESULT_DIR, out_name)
            cv2.imwrite(out_path, img)
            return {'path': out_path}

        # Create 3-frame sequence directory for ProPainter
        frame_dir = os.path.join(TEMP_DIR, f"image_frames_{unique_suffix}")
        mask_dir = os.path.join(PROPAINTER_MASK_ROOT, f"{base_name}_{unique_suffix}")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        self.update_state(state='PROCESSING', meta={'progress': 25, 'status': 'Preparing frames'})

        # Create 3 identical frames (ProPainter needs temporal context)
        for i in range(3):
            frame_path = os.path.join(frame_dir, f"{i:04d}.png")
            cv2.imwrite(frame_path, img)

        # Generate mask - precise detection like YOLO GOOD version
        mask = det.create_mask(img, detections)
        if mask is None or mask.size == 0:
            out_name = base_name + '_clean.png'
            out_path = os.path.join(RESULT_DIR, out_name)
            cv2.imwrite(out_path, img)
            shutil.rmtree(frame_dir, ignore_errors=True)
            shutil.rmtree(mask_dir, ignore_errors=True)
            return {'path': out_path}

        # Ensure binary mask
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255

        # Save same mask for all 3 frames
        for i in range(3):
            mask_path = os.path.join(mask_dir, f"{i:04d}.png")
            cv2.imwrite(mask_path, mask)

        self.update_state(state='PROCESSING', meta={'progress': 45, 'status': 'Running faster-propainter pipeline'})

        # Use direct faster-propainter pipeline instead of subprocess for 3x speedup
        pipeline_start = performance_checkpoint("faster-propainter Pipeline")
        try:
            # Import faster-propainter pipeline at runtime
            import sys
            faster_propainter_path = os.path.join(SCRIPT_DIR, 'faster-propainter-main')
            if faster_propainter_path not in sys.path:
                sys.path.insert(0, faster_propainter_path)
            from watermark import pipeline as faster_propainter_pipeline
            
            import torch
            use_fp16 = torch.cuda.is_available()
            
            print(f"🚀 Running direct faster-propainter pipeline: FP16={use_fp16} + neighbor_length=20 + ref_stride=10 + subvideo_length=80")
            
            # Direct pipeline call - TRUE FASTER-PROPAINTER SETTINGS
            faster_propainter_pipeline(
                video=frame_dir,                    # Input frame directory
                mask=mask_dir,                      # Mask directory  
                output=PROPAINTER_OUTPUT_ROOT,      # Output directory
                resize_ratio=1.0,                   # Keep original resolution for images
                mask_dilation=4,                    # faster-propainter: standard mask dilation
                ref_stride=10,                      # faster-propainter: reduce references
                neighbor_length=20,                 # faster-propainter: recommended neighbors for quality
                subvideo_length=80,                 # faster-propainter: standard subvideo length
                raft_iter=20,                       # faster-propainter: standard optical flow
                mode="video_inpainting",            # Standard inpainting mode
                save_frames=True,                   # Save individual frames
                fp16=use_fp16                       # Enable FP16 if available
            )
            
            print("✅ faster-propainter pipeline completed successfully")
            performance_checkpoint("faster-propainter Pipeline", pipeline_start)
            
        except Exception as e:
            print(f"❌ faster-propainter pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"faster-propainter pipeline failed: {e}")
        finally:
            # Clear GPU memory after processing
            clear_gpu_memory()

        # Extract middle frame (frame 0001 out of 0000, 0001, 0002)
        # Frame directory name from ProPainter is the basename of input dir
        propainter_output_name = os.path.basename(frame_dir)
        save_root = os.path.join(PROPAINTER_OUTPUT_ROOT, propainter_output_name)
        middle_frame_path = os.path.join(save_root, 'frames', '0001.png')

        if not os.path.exists(middle_frame_path):
            # Fallback: try frame 0000 if middle doesn't exist
            middle_frame_path = os.path.join(save_root, 'frames', '0000.png')

        if not os.path.exists(middle_frame_path):
            raise RuntimeError(f"ProPainter output frame not found: {middle_frame_path}")

        self.update_state(state='PROCESSING', meta={'progress': 85, 'status': 'Finalizing'})

        # Copy result to final location
        out_name = base_name + '_clean.png'
        out_path = os.path.join(RESULT_DIR, out_name)
        shutil.copy2(middle_frame_path, out_path)

        # Cleanup temporary directories
        try:
            shutil.rmtree(frame_dir, ignore_errors=True)
            shutil.rmtree(mask_dir, ignore_errors=True)
            shutil.rmtree(save_root, ignore_errors=True)
        except Exception as cleanup_exc:
            print(f"⚠️  Failed to cleanup temp directories: {cleanup_exc}")

        print(f"✅ Image processed with ProPainter FP16: {out_path}")

        # If running remotely and the API host is reachable via TUNNEL_URL,
        # upload the result back so the frontend can download from the API server.
        uploaded_path = None
        tunnel = os.getenv('TUNNEL_URL')
        if tunnel and os.getenv('UPLOAD_RESULT_BACK', '1') == '1':
            try:
                import requests
                upload_url = tunnel.rstrip('/') + '/api/upload-result'
                print(f"⬆️  Uploading result to API host: {upload_url}")
                with open(out_path, 'rb') as fp:
                    resp = requests.post(
                        upload_url,
                        headers={'ngrok-skip-browser-warning': 'true'},
                        files={'file': (os.path.basename(out_path), fp, 'image/png')},
                        timeout=60
                    )
                if resp.ok:
                    j = resp.json()
                    if j.get('status') == 'success' and j.get('result_url'):
                        uploaded_path = j['result_url']
                        print(f"✅ Uploaded result registered at: {uploaded_path}")
                else:
                    print(f"⚠️  Upload back failed: HTTP {resp.status_code}")
            except Exception as up_err:
                print(f"⚠️  Upload back error: {up_err}")

        return {'path': uploaded_path or out_path}
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.remove_video')
def process_video_task(self, video_path):
    """
    Background task for video watermark removal using YOLO + ProPainter.
    """
    try:
        self.update_state(
            state='STARTED',
            meta={'progress': 0, 'status': 'Loading YOLO detector'}
        )

        detector = get_detector()
        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing - see logs for details")

        # If running on a remote worker (e.g., Salad), the local path from the API host
        # won't exist. In that case, try to download the file from the API via TUNNEL_URL.
        if not os.path.exists(video_path):
            tunnel = os.getenv('TUNNEL_URL')
            try:
                from urllib.parse import urljoin
                import requests
            except Exception:
                tunnel = None

            if tunnel:
                # Handle Windows-style paths coming from the API host
                try:
                    from pathlib import PureWindowsPath
                    base_name = PureWindowsPath(video_path).name
                except Exception:
                    base_name = os.path.basename(video_path.replace('\\', '/'))

                download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                print(f"🌐 Video not found locally. Downloading from: {download_url}")
                try:
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=30)
                    r.raise_for_status()
                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    video_path = remote_cached
                    print(f"✅ Downloaded to: {video_path}")
                except Exception as dl_err:
                    print(f"❌ Failed to download video from API host: {dl_err}")
                    raise Exception(f"Video file not found and remote download failed: {base_name}")
            else:
                raise Exception(f"Video file not found: {video_path}")

        print(f"Opening video for ProPainter: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")

        self.update_state(
            state='PROCESSING',
            meta={'progress': 5, 'status': 'Scanning video frames'}
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        base_name = Path(video_path).stem
        unique_suffix = self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8]
        mask_dir = os.path.join(PROPAINTER_MASK_ROOT, f"{base_name}_{unique_suffix}")
        os.makedirs(mask_dir, exist_ok=True)

        zero_mask = np.zeros((height, width), dtype=np.uint8)
        last_valid_bbox = None
        frames_processed = 0
        frames_with_watermark = 0
        detect_interval = 3   # Detect every 3rd frame to catch movements
        warmup_frames = 60    # Detect every frame for first N frames or until first hit
        hit_found = False
        det_conf = 0.15       # Balanced confidence like YOLO GOOD version

        # Track bbox per frame for segmentation
        bboxes_per_frame = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            progress = 5 + int((frames_processed / max(total_frames, 1)) * 40)
            self.update_state(
                state='PROCESSING',
                meta={
                    'progress': progress,
                    'status': f'Building masks {frames_processed}/{total_frames}'
                }
            )

            # Only run YOLO detection every Nth frame, but always during warmup until first hit
            if (frames_processed % detect_interval == 0) or (not hit_found and frames_processed < warmup_frames):
                detections = detector.detect(frame, confidence_threshold=det_conf, padding=0)  # Precise detection like YOLO GOOD
                active_detections = detections
                actual_detections = actual_detections + 1 if 'actual_detections' in locals() else 1

                if detections:
                    frames_with_watermark += 1
                    primary = detections[0]
                    if primary.get('confidence', 0) >= det_conf:
                        last_valid_bbox = primary['bbox']
                    hit_found = True
                elif last_valid_bbox:
                    active_detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]
            else:
                # Reuse last detected bounding box for frames in between
                active_detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}] if last_valid_bbox else []

            if active_detections:
                mask = detector.create_mask(frame, active_detections)
                # Track bbox for segmentation
                bboxes_per_frame.append(active_detections[0]['bbox'])
            else:
                mask = zero_mask
                # No detection for this frame
                bboxes_per_frame.append(None)

            mask_path = os.path.join(mask_dir, f"{frames_processed:04d}.png")
            cv2.imwrite(mask_path, mask)
            frames_processed += 1

        cap.release()

        if frames_processed == 0:
            raise RuntimeError("No frames were processed - video may be corrupted")

        detections_run = actual_detections if 'actual_detections' in locals() else (frames_processed + detect_interval - 1) // detect_interval
        print(f"✅ Masks generated: {frames_processed} frames total")
        print(f"   🎯 YOLO detections: {detections_run}/{frames_processed} frames (detect_interval={detect_interval})")
        print(f"   💧 Frames with watermark: {frames_with_watermark}")

        # Detect position segments for smart cropping
        from segment_detector import detect_segments, merge_adjacent_segments, should_use_cropping

        segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
        if segments:
            segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
            print(f"📊 Detected {len(segments)} watermark position segments:")
            for i, (start, end, bbox) in enumerate(segments):
                duration = end - start + 1
                print(f"   Segment {i+1}: frames {start}-{end} ({duration} frames) bbox={bbox}")

        use_cropping = should_use_cropping(segments, width, height, min_speedup=5.0) if segments else False

        if use_cropping and segments:
            print("🚀 Smart cropping enabled - processing segments individually")
            import subprocess

            # Extract original frames once for merging later
            original_frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{unique_suffix}_originals")
            os.makedirs(original_frames_dir, exist_ok=True)

            print(f"📷 Extracting original frames for merging...")
            extract_cmd = [
                'ffmpeg', '-i', video_path,
                '-qscale:v', '2',
                '-start_number', '0',  # Start numbering at 0 to match frame indices
                os.path.join(original_frames_dir, '%04d.png')
            ]
            subprocess.run(extract_cmd, capture_output=True, check=True)

            segment_context = {
                'base_name': base_name,
                'unique_suffix': unique_suffix,
                'width': width,
                'height': height,
                'fps': fps,
                'mask_dir': mask_dir,
                'original_frames_dir': original_frames_dir,
            }

            # Phase 1: ProPainter processing (GPU-bound, parallel across multiple GPUs)
            segment_cleaned_dirs = [None] * len(segments)
            segment_temp_dirs = [None] * len(segments)

            # Determine number of GPUs available
            num_gpus = min(SEGMENT_WORKERS, len(segments))

            print(f"🎨 Phase 1: ProPainter processing ({len(segments)} segments across {num_gpus} GPUs)")
            print(f"   🚀 Each segment gets its own GPU for true parallel processing")
            self.update_state(
                state='PROCESSING',
                meta={'progress': 55, 'status': f'ProPainter processing ({num_gpus} GPUs)'}
            )

            if num_gpus > 1:
                # Parallel processing with GPU assignment
                completed = 0
                print(f"🔥 Processing {len(segments)} segments in parallel across {num_gpus} GPUs...")

                # Create process pool with GPU initialization
                with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                    # Submit all segments with GPU assignment
                    futures = {}
                    for seg_idx in range(len(segments)):
                        gpu_id = seg_idx % num_gpus  # Round-robin GPU assignment
                        # Initialize worker with GPU, then process segment
                        future = executor.submit(
                            lambda idx, gpu, seg, ctx: (_init_gpu_worker(gpu), _process_propainter_segment(idx, len(segments), seg, ctx))[1],
                            seg_idx,
                            gpu_id,
                            segments[seg_idx],
                            segment_context
                        )
                        futures[future] = seg_idx

                    try:
                        for future in as_completed(futures):
                            seg_idx = futures[future]
                            _, cleaned_dir, crop_info, temp_dirs = future.result()
                            segment_cleaned_dirs[seg_idx] = cleaned_dir
                            segment_temp_dirs[seg_idx] = temp_dirs
                            completed += 1
                            progress = 55 + int((completed / len(segments)) * 15)
                            self.update_state(
                                state='PROCESSING',
                                meta={'progress': progress, 'status': f'ProPainter {completed}/{len(segments)} complete'}
                            )
                            print(f"✅ ProPainter segment {seg_idx+1}/{len(segments)} complete ({completed}/{len(segments)} total)")
                    except Exception:
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        raise
            else:
                # Sequential fallback
                for seg_idx, segment in enumerate(segments):
                    _, cleaned_dir, crop_info, temp_dirs = _process_propainter_segment(seg_idx, len(segments), segment, segment_context)
                    segment_cleaned_dirs[seg_idx] = cleaned_dir
                    segment_temp_dirs[seg_idx] = temp_dirs
                    progress = 55 + int(((seg_idx + 1) / len(segments)) * 15)
                    self.update_state(
                        state='PROCESSING',
                        meta={'progress': progress, 'status': f'ProPainter segment {seg_idx+1}/{len(segments)}'}
                    )
                    print(f"✅ ProPainter segment {seg_idx+1}/{len(segments)} complete")

            if any(d is None for d in segment_cleaned_dirs):
                raise RuntimeError("One or more segments failed ProPainter processing")

            # Clear GPU memory after all ProPainter processing completes
            print("🧹 Clearing GPU memory after ProPainter phase...")
            clear_gpu_memory()

            # Phase 2: Video encoding (CPU-bound, parallel)
            max_workers = min(4, len(segments))  # Use up to 4 threads for encoding
            print(f"\n🎞️  Phase 2: Video encoding ({len(segments)} segments in parallel, {max_workers} workers)")
            self.update_state(
                state='PROCESSING',
                meta={'progress': 70, 'status': f'Encoding segments'}
            )

            segment_videos = [None] * len(segments)
            encode_context = {
                'base_name': base_name,
                'unique_suffix': unique_suffix,
                'fps': fps
            }

            if max_workers > 1:
                completed = 0
                print(f"🧵 Encoding {len(segments)} segments in parallel...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _encode_segment,
                            seg_idx,
                            len(segments),
                            segment_cleaned_dirs[seg_idx],
                            encode_context,
                            segment_temp_dirs[seg_idx]
                        ): seg_idx
                        for seg_idx in range(len(segments))
                    }
                    try:
                        for future in as_completed(futures):
                            seg_idx = futures[future]
                            _, seg_video_path = future.result()
                            segment_videos[seg_idx] = seg_video_path
                            completed += 1
                            progress = 70 + int((completed / len(segments)) * 10)
                            self.update_state(
                                state='PROCESSING',
                                meta={'progress': progress, 'status': f'Encoding {completed}/{len(segments)} complete'}
                            )
                            print(f"✅ Encoded segment {seg_idx+1}/{len(segments)} ({completed}/{len(segments)} total)")
                    except Exception:
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        raise
            else:
                for seg_idx in range(len(segments)):
                    _, seg_video_path = _encode_segment(
                        seg_idx,
                        len(segments),
                        segment_cleaned_dirs[seg_idx],
                        encode_context,
                        segment_temp_dirs[seg_idx]
                    )
                    segment_videos[seg_idx] = seg_video_path
                    progress = 70 + int(((seg_idx + 1) / len(segments)) * 10)
                    self.update_state(
                        state='PROCESSING',
                        meta={'progress': progress, 'status': f'Encoding segment {seg_idx+1}/{len(segments)}'}
                    )

            if any(path is None for path in segment_videos):
                raise RuntimeError("One or more segments failed encoding")

            # Concatenate all segments
            print(f"\n🎬 Concatenating {len(segment_videos)} segments...")
            self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Concatenating segments'})

            concat_list_file = os.path.join(TEMP_DIR, f"{base_name}_{unique_suffix}_concat.txt")
            with open(concat_list_file, 'w') as f:
                for seg_video in segment_videos:
                    f.write(f"file '{seg_video}'\n")

            temp_processed = os.path.join(RESULT_DIR, f"{base_name}_propainter_video.mp4")
            concat_cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_list_file,
                '-c', 'copy',
                temp_processed
            ]
            subprocess.run(concat_cmd, capture_output=True, check=True)

            # Cleanup
            shutil.rmtree(original_frames_dir, ignore_errors=True)
            for seg_video in segment_videos:
                os.remove(seg_video)
            os.remove(concat_list_file)

        else:
            # Fall back to full video processing
            if not use_cropping and segments:
                print("ℹ️  Smart cropping not beneficial for this video, using full frame processing")

            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            self.update_state(
                state='PROCESSING',
                meta={'progress': 55, 'status': 'Running faster-propainter pipeline'}
            )

            try:
                # Import faster-propainter pipeline at runtime
                import sys
                faster_propainter_path = os.path.join(SCRIPT_DIR, 'faster-propainter-main')
                if faster_propainter_path not in sys.path:
                    sys.path.insert(0, faster_propainter_path)
                from watermark import pipeline as faster_propainter_pipeline
                
                import torch
                use_fp16 = torch.cuda.is_available()
                
                # Dynamic optimization based on video resolution
                dynamic_subvideo, _ = get_dynamic_subvideo_length(width, height)
                
                print(f"🚀 Direct faster-propainter pipeline: resolution={width}x{height}, FP16={use_fp16}")
                print(f"   faster-propainter: neighbor_length=20 + ref_stride=10 + raft_iter=20 + subvideo_length=80 + flow_backend={PROPAINTER_FLOW_BACKEND}")
                
                # Direct pipeline call - TRUE FASTER-PROPAINTER SETTINGS
                faster_propainter_pipeline(
                    video=video_path,                   # Input video
                    mask=mask_dir,                      # Mask directory
                    output=PROPAINTER_OUTPUT_ROOT,      # Output directory
                    resize_ratio=1.0,                   # Keep original resolution
                    mask_dilation=4,                    # faster-propainter: standard mask dilation
                    ref_stride=10,                      # faster-propainter: reduce references
                    neighbor_length=20,                 # faster-propainter: recommended neighbors for quality
                    subvideo_length=80,                 # faster-propainter: standard subvideo length
                    raft_iter=20,                       # faster-propainter: standard optical flow
                    mode="video_inpainting",            # Standard inpainting
                    save_fps=fps,                       # Preserve original FPS
                    save_frames=False,                  # Only save video, not frames
                    fp16=use_fp16                       # FP16 if available
                )
                
                print("✅ faster-propainter pipeline completed successfully")
                
            except Exception as e:
                print(f"❌ faster-propainter pipeline failed: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"faster-propainter pipeline failed: {e}")
            finally:
                # Clear GPU memory after full video processing
                clear_gpu_memory()

            save_root = os.path.join(PROPAINTER_OUTPUT_ROOT, base_name)
            produced_video = os.path.join(save_root, 'inpaint_out.mp4')
            if not os.path.exists(produced_video):
                raise RuntimeError(f"ProPainter output not found: {produced_video}")

            temp_processed = os.path.join(RESULT_DIR, f"{base_name}_propainter_video.mp4")
            shutil.copy2(produced_video, temp_processed)

        self.update_state(
            state='PROCESSING',
            meta={'progress': 85, 'status': 'Merging audio'}
        )

        final_output = os.path.join(RESULT_DIR, f"{base_name}_propainter.mp4")

        try:
            check_audio_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]

            has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
            has_audio = 'audio' in has_audio_check.stdout

            if has_audio:
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', temp_processed,
                    '-i', video_path,
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-c:v', 'copy',         # Don't re-encode, just copy video stream!
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    final_output
                ]
            else:
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', temp_processed,
                    '-c:v', 'copy',         # No audio, just copy video stream
                    final_output
                ]

            print(f"Running FFmpeg audio merge: {' '.join(cmd)}")
            audio_merge = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if audio_merge.returncode != 0:
                print("⚠️  Audio merge failed, returning video without audio")
                print(audio_merge.stderr)
                final_output = temp_processed
            else:
                if has_audio:
                    verify_cmd = [
                        'ffprobe',
                        '-v', 'error',
                        '-select_streams', 'a:0',
                        '-show_entries', 'stream=codec_type',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        final_output
                    ]
                    verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)
                    if 'audio' not in verify.stdout:
                        print("⚠️  Audio verification failed, keeping silent video")
                        final_output = temp_processed
                if final_output != temp_processed and os.path.exists(temp_processed):
                    os.remove(temp_processed)
        except FileNotFoundError:
            print("⚠️  ffmpeg/ffprobe not available, returning silent video")
            final_output = temp_processed
        except subprocess.TimeoutExpired:
            print("⚠️  FFmpeg timed out, returning silent video")
            final_output = temp_processed

        try:
            shutil.rmtree(mask_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            print(f"⚠️  Failed to delete mask directory {mask_dir}: {cleanup_exc}")

        # Note: Don't call self.update_state(state='SUCCESS') - it overrides the return value!
        # Celery automatically sets state to SUCCESS when the task returns normally

        # If running remotely and the API host is reachable via TUNNEL_URL,
        # upload the result back so the frontend can download from the API server.
        uploaded_path = None
        tunnel = os.getenv('TUNNEL_URL')
        if tunnel and os.getenv('UPLOAD_RESULT_BACK', '1') == '1':
            try:
                import requests
                upload_url = tunnel.rstrip('/') + '/api/upload-result'
                print(f"⬆️  Uploading result to API host: {upload_url}")
                with open(final_output, 'rb') as fp:
                    resp = requests.post(
                        upload_url,
                        headers={'ngrok-skip-browser-warning': 'true'},
                        files={'file': (os.path.basename(final_output), fp, 'video/mp4')},
                        timeout=60
                    )
                if resp.ok:
                    j = resp.json()
                    if j.get('status') == 'success' and j.get('result_url'):
                        uploaded_path = j['result_url']  # e.g. /results/<filename>
                        print(f"✅ Uploaded result registered at: {uploaded_path}")
                else:
                    print(f"⚠️  Upload back failed: HTTP {resp.status_code}")
            except Exception as up_err:
                print(f"⚠️  Upload back error: {up_err}")

        return {
            'path': uploaded_path or final_output,
            'metadata': {
                'frames_processed': frames_processed,
                'frames_with_watermark': frames_with_watermark,
                'fps': fps,
                'width': width,
                'height': height,
                'propainter_output': final_output,
            }
        }

    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        raise



# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/ads.txt')
def ads_txt():
    """Serve ads.txt file for Google AdSense"""
    return send_file(os.path.join(app.static_folder, 'ads.txt'), mimetype='text/plain')


@app.route('/tunnel_url.txt')
def tunnel_url():
    """Serve tunnel URL for frontend auto-detection"""
    # Check environment variable first (for Railway deployment)
    env_url = os.getenv('TUNNEL_URL')
    if env_url:
        return env_url, 200, {'Content-Type': 'text/plain'}

    # Check file (for local development with localtunnel)
    tunnel_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
    if os.path.exists(tunnel_file):
        return send_file(tunnel_file, mimetype='text/plain')

    # Fallback to localhost
    return "http://localhost:9000", 200, {'Content-Type': 'text/plain'}


@app.route('/')
def index():
    """Serve landing page"""
    return send_file('web/index.html')


@app.route('/web/<path:path>')
def serve_web(path):
    """Serve static files"""
    return send_file(f'web/{path}')



@app.route('/api/remove-watermark', methods=['POST'])
def remove_watermark():
    """
    Submit image for watermark removal

    Returns:
        {
            'task_id': str,
            'status': 'queued'
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read file
        image_data = file.read()

        # Calculate hash for caching/deduplication
        file_hash = hashlib.md5(image_data).hexdigest()

        # Determine file type
        is_video = file.filename.lower().endswith(('.mp4', '.mov', '.avi'))

        if is_video:
            # Save video temporarily (D drive only!)
            temp_filename = f"{file_hash}{os.path.splitext(file.filename)[1]}"
            temp_path = os.path.join(UPLOAD_DIR, temp_filename)

            with open(temp_path, 'wb') as f:
                f.write(image_data)

            # Queue video processing
            task = process_video_task.apply_async(args=[temp_path])

            return jsonify({
                'task_id': task.id,
                'status': 'queued',
                'file_type': 'video'
            })

        # Process image with YOLO + ProPainter
        # Save image temporarily
        temp_filename = f"{file_hash}.png"
        temp_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_path, 'wb') as f:
            f.write(image_data)

        # Queue image processing with ProPainter
        task = process_image_task.apply_async(args=[temp_path])

        return jsonify({
            'task_id': task.id,
            'status': 'queued',
            'file_type': 'image'
        })

    except Exception as e:
        print(f"❌ Error queuing task: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<task_id>', methods=['GET', 'OPTIONS'])
def get_status(task_id):
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Check task status

    Returns:
        {
            'state': 'PENDING' | 'STARTED' | 'PROCESSING' | 'SUCCESS' | 'FAILURE',
            'progress': str,
            'result': { 'result_url': str } (if SUCCESS)
        }
    """
    try:
        # Special handling for distributed tasks (manual Redis tracking)
        if task_id.startswith('distributed_'):
            video_id = task_id.replace('distributed_', '')
            tracking_key = f"segments:{video_id}"

            # Check if tracking exists
            total_bytes = celery.backend.get(f"{tracking_key}:total")
            if not total_bytes:
                return jsonify({
                    'state': 'PENDING',
                    'progress': 'Task is waiting in queue...',
                    'info': {'progress': 0, 'status': 'Waiting in queue...'}
                })

            # Get completion status
            completed_bytes = celery.backend.get(tracking_key) or b'0'
            total = int(total_bytes)
            completed = int(completed_bytes)

            print(f"📊 Distributed task status - {task_id}: {completed}/{total} segments complete")

            if completed < total:
                # Still processing segments
                progress = int((completed / total) * 90)  # 0-90%
                return jsonify({
                    'state': 'PROCESSING',
                    'progress': f'Processing segments: {completed}/{total}',
                    'info': {'progress': progress, 'status': f'Segment {completed}/{total} complete'}
                })
            else:
                # All segments done, finalize should be complete or completing
                # Get base_name from prepare_result to construct correct filename
                import json
                prepare_result_json = celery.backend.get(f"{tracking_key}:prepare_result")
                if prepare_result_json:
                    prepare_result = json.loads(prepare_result_json)
                    base_name = prepare_result.get('base_name', video_id)
                    result_filename = f"{base_name}_propainter.mp4"
                else:
                    # Fallback if prepare_result not found
                    result_filename = f"{video_id}_propainter.mp4"

                return jsonify({
                    'state': 'SUCCESS',
                    'result': {'result_url': f'/results/{result_filename}'},
                    'metadata': {'total_segments': total}
                })

        # Regular Celery task handling
        from celery.result import AsyncResult

        task = AsyncResult(task_id, app=celery)

        # Always log for debugging stuck "Waiting in queue" issue
        print(f"📊 Status check - Task: {task_id}, State: {task.state}, Info: {task.info}")

        response = {
            'state': task.state
        }

        if task.state == 'PENDING':
            response['progress'] = 'Task is waiting in queue...'
            response['info'] = {'progress': 0, 'status': 'Waiting in queue...'}
        elif task.state == 'STARTED':
            info = task.info or {}
            response['progress'] = info.get('status', 'Starting...')
            response['info'] = {'progress': info.get('progress', 5), 'status': info.get('status', 'Starting...')}
        elif task.state == 'PROCESSING':
            info = task.info or {}
            response['progress'] = info.get('status', 'Processing...')
            response['info'] = {'progress': info.get('progress', 50), 'status': info.get('status', 'Processing...')}
        elif task.state == 'SUCCESS':
            # task.result is a dict with 'path' and 'metadata' OR 'chord_id' for distributed tasks
            result_data = task.result
            print(f"🔍 DEBUG - task.result type: {type(result_data)}, value: {result_data}")

            if isinstance(result_data, dict):
                # Check if this is a prepare task that returned chord_id
                if 'chord_id' in result_data:
                    chord_id = result_data['chord_id']
                    print(f"🔄 Prepare task complete, switching to chord tracking: {chord_id}")
                    # Tell frontend to switch to tracking the chord instead
                    # Frontend expects HTTP 409 status code to detect task switching
                    response['state'] = 'PROCESSING'
                    response['progress'] = result_data.get('message', 'Processing segments in parallel')
                    response['info'] = {
                        'progress': 50,
                        'status': 'Segments processing in parallel'
                    }
                    response['stale'] = True  # Frontend recognizes this pattern
                    response['current_task_id'] = chord_id  # Frontend will switch to this task_id
                    return jsonify(response), 409  # HTTP 409 Conflict - task superseded

                result_path = result_data.get('path')
                if not result_path:
                    print(f"❌ Task {task_id} SUCCESS but no path in result: {result_data}")
                    return jsonify({'error': 'Invalid result format - missing path'}), 500
            else:
                result_path = result_data

            # Final safety check for None result_path
            if not result_path:
                print(f"❌ Task {task_id} has None result_path")
                return jsonify({'error': 'Invalid result - path is None'}), 500

            filename = os.path.basename(result_path)
            response['result'] = {
                'result_url': f'/results/{filename}'
            }
            if isinstance(result_data, dict) and 'metadata' in result_data:
                response['metadata'] = result_data['metadata']
        elif task.state == 'FAILURE':
            response['error'] = str(task.info)
            print(f"❌ Task failed: {task.info}")

        return jsonify(response)
    except Exception as e:
        print(f"❌ Status endpoint error for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500


@app.route('/api/result/<task_id>', methods=['GET'])
def get_result(task_id):
    """
    Download processed file

    Returns:
        Processed image/video file
    """
    from celery.result import AsyncResult

    task = AsyncResult(task_id, app=celery)

    if task.state != 'SUCCESS':
        return jsonify({
            'error': 'Task not complete',
            'status': task.state
        }), 400

    try:
        result = task.result

        if 'data' in result:
            # Image result
            result_bytes = result['data']
            return send_file(
                io.BytesIO(result_bytes),
                mimetype='image/png',
                as_attachment=True,
                download_name='watermark_removed.png'
            )
        elif 'path' in result:
            # Video result
            return send_file(
                result['path'],
                mimetype='video/mp4',
                as_attachment=True,
                download_name='watermark_removed.mp4'
            )
        else:
            return jsonify({'error': 'Invalid result format'}), 500

    except Exception as e:
        print(f"❌ Error retrieving result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-from-url', methods=['POST', 'OPTIONS'])
def download_from_url():
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Download video from URL using Playwright (bypasses Cloudflare)
    Works for most video sites with anti-bot protection

    Request: { "url": "https://..." }
    Response: { "status": "success", "task_id": "...", "video_url": "/uploads/..." }
    """
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

        # Normalize URL - handle partial URLs from ChatGPT Sora
        url = url.strip()

        # If URL starts with /backend or is a path, prepend Sora domain
        if url.startswith('/') or not url.startswith('http'):
            if 'sora' in url or 'backend/project_y' in url:
                url = 'https://sora.chatgpt.com' + (url if url.startswith('/') else '/' + url)
            else:
                return jsonify({'status': 'error', 'message': 'URL must start with http:// or https://'}), 400

        print(f"📋 Normalized URL: {url}")

        # Validate URL to prevent SSRF attacks
        if not validate_url(url):
            return jsonify({'status': 'error', 'message': 'Invalid or unsafe URL'}), 400

        # Generate unique filename
        task_id = str(uuid.uuid4())
        output_path = os.path.join(UPLOAD_DIR, f'{task_id}.mp4')

        # Use Playwright for better Cloudflare bypass
        from playwright.sync_api import sync_playwright
        import time
        import re
        import html

        with sync_playwright() as p:
            print("🚀 Launching browser for video download...")
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )

            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                viewport={'width': 1920, 'height': 1080}
            )

            page = context.new_page()

            # Hide webdriver
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            print(f"🌐 Navigating to: {url}")
            page.goto(url, wait_until='domcontentloaded', timeout=30000)

            time.sleep(2)

            print("🔍 Extracting video URL...")

            # Collect network requests for video URLs
            video_urls = []

            def handle_response(response):
                if '.mp4' in response.url or 'video' in response.headers.get('content-type', ''):
                    video_urls.append(response.url)

            page.on('response', handle_response)

            # Wait for network activity
            time.sleep(2)

            # Find video URL in page content
            content = page.content()
            content_video_urls = re.findall(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content)
            video_urls.extend(content_video_urls)

            if not video_urls:
                # Try to find video element
                try:
                    video_element = page.locator('video').first
                    video_src = video_element.get_attribute('src')
                    if video_src:
                        video_urls = [video_src]
                    else:
                        # Try source elements
                        sources = page.locator('video source').all()
                        for source in sources:
                            src = source.get_attribute('src')
                            if src:
                                video_urls.append(src)
                except Exception as e:
                    print(f"⚠️  Error checking video element: {e}")

            if video_urls:
                video_src = html.unescape(video_urls[0])
                print(f"✅ Found video URL: {video_src}")

                # Download the video
                import requests
                response = requests.get(video_src, stream=True, timeout=300)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                browser.close()
                print(f"✅ Video downloaded: {output_path}")

                return jsonify({
                    'status': 'success',
                    'task_id': task_id,
                    'video_url': f'/uploads/{task_id}.mp4'
                })
            else:
                # Save debug screenshot
                screenshot_path = os.path.join(TEMP_DIR, f'debug_{task_id}.png')
                page.screenshot(path=screenshot_path)
                print(f"📸 Debug screenshot saved: {screenshot_path}")

                # Save page HTML for debugging
                debug_html_path = os.path.join(TEMP_DIR, f'debug_{task_id}.html')
                with open(debug_html_path, 'w', encoding='utf-8') as f:
                    f.write(page.content())
                print(f"📄 Debug HTML saved: {debug_html_path}")

                browser.close()
                return jsonify({
                    'status': 'error',
                    'message': 'Could not find video source URL. Debug files saved.',
                    'debug': {
                        'screenshot': f'/temp/debug_{task_id}.png',
                        'html': f'/temp/debug_{task_id}.html'
                    }
                }), 404

    except Exception as e:
        print(f"❌ Download error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/download-sora', methods=['POST', 'OPTIONS'])
def download_sora():
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Download Sora video from OpenAI using Playwright bypass + cookies
    Bypasses Cloudflare protection using saved cookies

    Request: { "url": "https://sora.chatgpt.com/..." or "/backend/project_y/..." }
    Response: { "status": "success", "task_id": "...", "video_url": "/uploads/..." }
    """
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

        # Normalize URL - handle partial URLs from ChatGPT Sora
        url = url.strip()

        # If URL starts with /backend or is a path, prepend Sora domain
        if url.startswith('/') or not url.startswith('http'):
            url = 'https://sora.chatgpt.com' + (url if url.startswith('/') else '/' + url)

        print(f"📋 Normalized Sora URL: {url}")

        # Validate URL to prevent SSRF attacks
        if not validate_url(url):
            return jsonify({'status': 'error', 'message': 'Invalid or unsafe URL'}), 400

        # Generate unique filename
        task_id = str(uuid.uuid4())
        output_path = os.path.join(UPLOAD_DIR, f'{task_id}.mp4')

        # Import Playwright
        from playwright.sync_api import sync_playwright
        import time
        import json

        # Path to cookies file
        cookies_file = os.path.join(SCRIPT_DIR, 'downz', 'cookies.json')
        print(f"🔍 Looking for cookies at: {cookies_file}")
        print(f"🔍 SCRIPT_DIR is: {SCRIPT_DIR}")

        with sync_playwright() as p:
            print("🚀 Launching browser for Sora download...")
            browser = p.chromium.launch(
                headless=True,  # Run headless in production
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )

            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York'
            )

            # Load cookies if they exist
            if os.path.exists(cookies_file):
                print(f"📂 Loading cookies from {cookies_file}...")
                with open(cookies_file, 'r') as f:
                    cookies = json.load(f)
                    context.add_cookies(cookies)
                print("✅ Cookies loaded!")
            else:
                print(f"⚠️  No cookies found at: {cookies_file}")
                browser.close()
                return jsonify({
                    'status': 'error',
                    'message': 'Authentication required for Sora videos. Please contact administrator to set up cookies.',
                    'hint': 'Sora videos require login cookies from ChatGPT.'
                }), 401

            page = context.new_page()

            # Hide webdriver
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            print(f"🌐 Navigating to: {url}")
            page.goto(url, wait_until='domcontentloaded', timeout=30000)

            time.sleep(2)

            print("🔍 Extracting video URL from page content...")

            video_src = None

            # Try to find video URLs in page content (works for Sora)
            content = page.content()
            import re
            video_urls = re.findall(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content)

            if video_urls:
                import html
                video_src = html.unescape(video_urls[0])  # Decode &amp; to &
                print(f"✅ Found video URL in page content: {video_src}")
            else:
                # Fallback: try video element
                print("⏳ No .mp4 URL found, trying video element...")
                video_element = page.query_selector('video')
                if video_element:
                    video_src = video_element.get_attribute('src')
                    if not video_src:
                        source = page.query_selector('video source')
                        if source:
                            video_src = source.get_attribute('src')

                if not video_src:
                    # Take screenshot for debugging
                    screenshot_path = os.path.join(TEMP_DIR, f'debug_{task_id}.png')
                    page.screenshot(path=screenshot_path)
                    print(f"📸 Screenshot saved to {screenshot_path}")

                    browser.close()
                    return jsonify({
                        'status': 'error',
                        'message': f'No video URL found. Screenshot saved to {screenshot_path}'
                    }), 404

            if video_src:
                print(f"✅ Found video source: {video_src}")

                # Make absolute URL if relative
                if video_src.startswith('//'):
                    video_src = 'https:' + video_src
                elif video_src.startswith('/'):
                    from urllib.parse import urljoin
                    video_src = urljoin(url, video_src)

                print(f"⬇️  Downloading video...")

                # Download using Playwright's request context with retry logic
                max_retries = 3
                retry_delay = 3

                for retry in range(max_retries):
                    try:
                        if retry > 0:
                            print(f"🔄 Retry attempt {retry}/{max_retries-1}...")
                            time.sleep(retry_delay)
                        else:
                            # Small delay before first attempt to avoid rate limiting
                            time.sleep(1)

                        response = page.request.get(video_src)

                        if response.ok:
                            with open(output_path, 'wb') as f:
                                f.write(response.body())

                            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                            print(f"✅ Downloaded successfully! Size: {file_size:.2f} MB")

                            browser.close()

                            return jsonify({
                                'status': 'success',
                                'task_id': task_id,
                                'video_url': f'/uploads/{task_id}.mp4'
                            })
                        else:
                            print(f"⚠️  Download failed: HTTP {response.status}")
                            if retry == max_retries - 1:
                                browser.close()
                                return jsonify({
                                    'status': 'error',
                                    'message': f'Download failed after {max_retries} attempts: HTTP {response.status}'
                                }), 500
                    except Exception as download_error:
                        print(f"⚠️  Download error: {download_error}")
                        if retry == max_retries - 1:
                            browser.close()
                            return jsonify({
                                'status': 'error',
                                'message': f'Download failed after {max_retries} attempts: {str(download_error)}'
                            }), 500
            else:
                browser.close()
                return jsonify({
                    'status': 'error',
                    'message': 'Could not find video source URL'
                }), 404

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def check_rate_limit(ip):
    """Check if IP has exceeded rate limit (10 uploads per hour)"""
    current_time = time.time()

    if ip in UPLOAD_RATE_LIMIT:
        count, first_upload_time = UPLOAD_RATE_LIMIT[ip]

        # Reset if hour has passed
        if current_time - first_upload_time > 3600:
            UPLOAD_RATE_LIMIT[ip] = (1, current_time)
            return True

        # Check limit
        if count >= 10:
            return False

        UPLOAD_RATE_LIMIT[ip] = (count + 1, first_upload_time)
        return True
    else:
        UPLOAD_RATE_LIMIT[ip] = (1, current_time)
        return True

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Upload video/image file (rate limited to 10/hour per IP)

    Returns: { "status": "success", "task_id": "uuid" }
    """
    try:
        # Rate limiting
        client_ip = request.remote_addr
        if not check_rate_limit(client_ip):
            return jsonify({
                'status': 'error',
                'message': 'Rate limit exceeded. Maximum 10 uploads per hour.'
            }), 429

        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Get file extension
        ext = os.path.splitext(file.filename)[1] or '.mp4'

        # Save file
        file_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
        file.save(file_path)

        print(f"✅ File uploaded: {file_path}")

        return jsonify({
            'status': 'success',
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/process', methods=['POST', 'OPTIONS'])
def process_video():
    if request.method == 'OPTIONS':
        return ('', 204)
    """
    Process video to remove watermarks

    Request: { "task_id": "uuid-from-download" }
    Response: { "status": "success", "task_id": "celery-task-id" }
    """
    try:
        data = request.get_json()
        task_id = data.get('task_id')

        if not task_id:
            return jsonify({'status': 'error', 'message': 'No task_id provided'}), 400

        # Find uploaded media (video first, then image) with any extension
        video_path = None
        # Try common video extensions
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
            if os.path.exists(test_path):
                video_path = test_path
                break
        # If not found, try common image extensions
        if not video_path:
            for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
                if os.path.exists(test_path):
                    video_path = test_path
                    break

        if not video_path:
            return jsonify({'status': 'error', 'message': 'Media not found'}), 404

        # Queue processing task via Celery and return the real task id
        print(f"📤 Queuing processing task for: {video_path}")

        try:
            # Decide pipeline based on extension
            ext = os.path.splitext(video_path)[1].lower()
            image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            if ext in image_exts:
                result = celery.send_task('watermark.remove_image', args=[video_path])
            else:
                # Use distributed processing for videos (multiple workers collaborate on segments)
                # Build a Celery canvas chain on the server side to avoid any in-task blocking
                def _current_public_base():
                    env_url = os.getenv('TUNNEL_URL')
                    if env_url:
                        return env_url.strip()
                    try:
                        tunnel_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
                        if os.path.exists(tunnel_file):
                            with open(tunnel_file, 'r') as f:
                                return f.read().strip()
                    except Exception:
                        pass
                    return 'http://localhost:9000'

                base = _current_public_base()
                # Call prepare_video_task which creates the chord internally
                # This creates: prepare -> chord([segment1, segment2, ...]) -> finalize
                result = prepare_video_task.apply_async(
                    args=[video_path],
                    kwargs={'api_base': base, 'temp_base': base}
                )
            print(f"✅ Task queued with ID: {result.id}")
            return jsonify({'status': 'success', 'task_id': result.id})

        except Exception as e:
            print(f"❌ Failed to queue task: {e}")
            import traceback; traceback.print_exc()
            return jsonify({'status': 'error', 'message': f'Failed to connect to Redis: {str(e)}'}), 500

    except Exception as e:
        print(f"❌ Process endpoint error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve uploaded video files"""
    # Sanitize filename to prevent path traversal
    filename = sanitize_filename(filename)
    file_path = os.path.join(UPLOAD_DIR, filename)

    # Verify file exists and is within upload directory
    if not os.path.exists(file_path) or not os.path.abspath(file_path).startswith(os.path.abspath(UPLOAD_DIR)):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path)


@app.route('/results/<filename>')
def serve_result(filename):
    """Serve processed result files and delete after sending"""
    # Sanitize filename to prevent path traversal
    filename = sanitize_filename(filename)
    file_path = os.path.join(RESULT_DIR, filename)

    # Verify file exists and is within result directory
    if not os.path.exists(file_path) or not os.path.abspath(file_path).startswith(os.path.abspath(RESULT_DIR)):
        return jsonify({'error': 'File not found'}), 404

    # Send file with as_attachment to trigger download
    response = send_file(file_path, as_attachment=True, download_name=f'cleaned_{filename}')

    # Schedule file deletion after response is sent
    @response.call_on_close
    def delete_files():
        try:
            # Delete the result file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  Deleted result: {filename}")

            # Delete corresponding upload file
            # Extract original task_id from processed filename
            original_name = filename.replace('_processed.avi', '.mp4')
            upload_path = os.path.join(UPLOAD_DIR, original_name)
            if os.path.exists(upload_path):
                os.remove(upload_path)
                print(f"🗑️  Deleted upload: {original_name}")
        except Exception as e:
            print(f"⚠️  Error deleting files: {e}")

    return response


@app.route('/api/upload-result', methods=['POST', 'OPTIONS'])
def upload_result():
    if request.method == 'OPTIONS':
        return ('', 204)
    """Accept result file from a remote worker and store it under results/.

    Request: multipart/form-data with field 'file' and optional 'filename'.
    Response: { "status": "success", "result_url": "/results/<filename>" }
    """
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        up = request.files['file']
        if up.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        req_filename = request.form.get('filename') or up.filename
        safe_name = sanitize_filename(req_filename)
        os.makedirs(RESULT_DIR, exist_ok=True)
        dest = os.path.join(RESULT_DIR, safe_name)
        up.save(dest)
        return jsonify({'status': 'success', 'result_url': f'/results/{safe_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/upload-segment', methods=['POST', 'OPTIONS'])
def upload_segment():
    """
    Upload a processed segment video from a distributed worker.
    Used by process_segment_task to upload completed segment videos.

    Request: multipart/form-data with 'file', 'video_id', 'seg_idx'
    Response: { "status": "success", "segment_url": "/results/<filename>" }
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        video_id = request.form.get('video_id', '')
        seg_idx = request.form.get('seg_idx', '')

        safe_name = sanitize_filename(file.filename)
        os.makedirs(RESULT_DIR, exist_ok=True)
        dest = os.path.join(RESULT_DIR, safe_name)
        file.save(dest)

        print(f"✅ Received segment upload: video_id={video_id}, seg_idx={seg_idx}, file={safe_name}")

        return jsonify({
            'status': 'success',
            'segment_url': f'/results/{safe_name}',
            'video_id': video_id,
            'seg_idx': seg_idx
        })
    except Exception as e:
        print(f"❌ Segment upload error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/temp/<path:filepath>')
def serve_temp_file(filepath):
    """
    Serve temporary files for distributed workers.
    Allows workers to download shared frames, masks, etc.

    Security: Only serve from TEMP_DIR, block path traversal
    """
    try:
        # Sanitize path to prevent directory traversal
        safe_path = os.path.normpath(filepath).lstrip('/')
        full_path = os.path.join(TEMP_DIR, safe_path)

        # Verify path is within TEMP_DIR
        if not os.path.abspath(full_path).startswith(os.path.abspath(TEMP_DIR)):
            return jsonify({'error': 'Access denied'}), 403

        if not os.path.exists(full_path):
            return jsonify({'error': 'File not found'}), 404

        if os.path.isfile(full_path):
            return send_file(full_path)
        else:
            return jsonify({'error': 'Not a file'}), 400

    except Exception as e:
        print(f"❌ Error serving temp file {filepath}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/privacy')
def privacy_policy():
    """Serve Privacy Policy page"""
    return send_file(os.path.join(app.static_folder, 'privacy.html'))


@app.route('/terms')
def terms_of_service():
    """Serve Terms of Service page"""
    return send_file(os.path.join(app.static_folder, 'terms.html'))


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get server statistics

    Returns:
        {
            'queue_length': int,
            'active_tasks': int,
            'completed_today': int
        }
    """
    # Get Celery stats
    from celery.task.control import inspect

    i = inspect(app=celery)
    active = i.active()
    scheduled = i.scheduled()

    active_count = sum(len(tasks) for tasks in (active or {}).values())
    scheduled_count = sum(len(tasks) for tasks in (scheduled or {}).values())

    return jsonify({
        'queue_length': scheduled_count,
        'active_tasks': active_count,
        'timestamp': datetime.utcnow().isoformat()
    })




# ============================================================================
# Run Server
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("WatermarkAI Production Server")
    print("=" * 60)
    print("Starting Flask server on http://0.0.0.0:9000")
    print("")
    print("To start Celery worker (in separate terminal):")
    print("  celery -A server_production.celery worker --loglevel=info --concurrency=2")
    print("")
    print("To start Redis (required):")
    print("  redis-server")
    print("")
    print("=" * 60)

    # Run Flask app
    app.run(
        host='0.0.0.0',  # Listen on all interfaces for tunnel
        port=9000,
        debug=False,  # Set to False for production
        threaded=True
    )
# Explicit CORS preflight catch-all for /api/* (helps when proxies strip headers)
@app.route('/api/<path:subpath>', methods=['OPTIONS'])
def cors_preflight(subpath):
    return ('', 204)
