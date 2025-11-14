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

# Load environment variables from .env file (for Celery Redis configuration)
from dotenv import load_dotenv
load_dotenv()

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

# Force parallel segmentation for multi-GPU/multi-worker distribution (only if YOLO fails)
os.environ.setdefault('MIN_SEGMENTS', '5')  # Fallback split if YOLO finds 0-1 segments
os.environ.setdefault('MIN_CHUNK_FRAMES', '60')  # Minimum frames per chunk (fallback when YOLO fails)

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
        print("[WARNING]  CUDA not detected in system torch; continuing with CPU mode.")
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


# Try to initialize CUDA/torch (local workers only)
# Railway deployment runs API-only mode (no GPU needed)
GPU_AVAILABLE = False
try:
    _ensure_cuda_torch()
    GPU_AVAILABLE = True
    print("[OK] GPU/CUDA initialized successfully - worker mode enabled")
except Exception as e:
    print(f"[INFO] Running in API-only mode (no GPU): {e}")
    print("[INFO] This is normal for Railway deployment - local workers handle GPU processing")

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from celery import Celery, chord

# Conditional imports for GPU processing (only needed on local workers)
try:
    import cv2
except ImportError:
    cv2 = None
    if not GPU_AVAILABLE:
        print("[INFO] cv2 not available (API-only mode)")

import numpy as np
import io
import json
import time
import hashlib
import uuid
from datetime import datetime
import redis
import threading
import secrets
import hmac
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# [INIT] FFmpeg/FFprobe path detection with fallback
def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths with fallback to static-ffmpeg."""
    # Try system PATH first (best performance)
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')

    if ffmpeg_path and ffprobe_path:
        print(f"[OK] Using system FFmpeg: {ffmpeg_path}")
        return ffmpeg_path, ffprobe_path

    # Fallback to static-ffmpeg (includes BOTH ffmpeg and ffprobe!)
    try:
        from static_ffmpeg import run
        ffmpeg_path, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
        print(f"[OK] Using static-ffmpeg: {ffmpeg_path}")
        print(f"[OK] FFprobe available: {ffprobe_path}")
        return ffmpeg_path, ffprobe_path
    except ImportError:
        print("[ERROR] static-ffmpeg not installed and no system FFmpeg found")
        raise RuntimeError("FFmpeg/FFprobe not available. Install via: pip install static-ffmpeg")
    except Exception as e:
        print(f"[ERROR] Failed to get static-ffmpeg executables: {e}")
        raise RuntimeError(f"FFmpeg/FFprobe initialization failed: {e}")

# Initialize FFmpeg paths at module level (before Celery workers start)
# FFmpeg only needed on local workers (video processing), not on Railway API
try:
    FFMPEG_EXE, FFPROBE_EXE = get_ffmpeg_executables()
except Exception as e:
    FFMPEG_EXE, FFPROBE_EXE = None, None
    if not GPU_AVAILABLE:
        print(f"[INFO] FFmpeg not available (API-only mode): {e}")

# [INIT] EXTREME SPEED: Global in-memory frame/mask cache
# Shared across all threads in Celery worker (threads pool)
# Stores frames/masks in RAM for instant access (no Redis, no disk!)
FRAME_CACHE = {}
FRAME_CACHE_LOCK = threading.Lock()

# Import faster-propainter modules for direct pipeline processing
# Note: These imports are moved to inside functions to avoid startup errors
# sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'faster-propainter-main'))
# from watermark import pipeline as faster_propainter_pipeline
# from mytimer import timer_decorator  
# from pre_post_process import crop_video_mask, merge_videos_with_mask

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web'))
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

@app.route('/api/debug/files', methods=['GET'])
def debug_files():
    """Debug endpoint to check what files exist in web/ folder"""
    try:
        import glob
        web_files = []
        static_folder = app.static_folder

        # List files in web/ directory
        if os.path.exists(static_folder):
            for root, dirs, files in os.walk(static_folder):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), static_folder)
                    web_files.append(rel_path)

        return jsonify({
            'static_folder': static_folder,
            'static_folder_exists': os.path.exists(static_folder),
            'cwd': os.getcwd(),
            'script_dir': os.path.dirname(os.path.abspath(__file__)),
            'web_files_count': len(web_files),
            'web_files': web_files[:50],  # First 50 files
            'config_exists': os.path.exists(os.path.join(static_folder, 'config.js')),
            'index_exists': os.path.exists(os.path.join(static_folder, 'index.html')),
            'emblem_exists': os.path.exists(os.path.join(static_folder, 'emblem.png'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/celery', methods=['GET'])
def debug_celery():
    """Debug endpoint to check Celery/Redis configuration"""
    try:
        # Mask sensitive passwords in URLs
        def mask_url(url):
            if not url:
                return None
            import re
            return re.sub(r'(redis://[^:]+:)([^@]+)(@)', r'\1****\3', url)

        return jsonify({
            'redis_url_from_file': mask_url(REDIS_URL),
            'celery_broker_env': mask_url(os.getenv('CELERY_BROKER_URL')),
            'celery_backend_env': mask_url(os.getenv('CELERY_RESULT_BACKEND')),
            'celery_broker_config': mask_url(app.config.get('broker_url')),
            'celery_backend_config': mask_url(app.config.get('result_backend')),
            'celery_broker_actual': mask_url(celery.conf.broker_url),
            'celery_backend_actual': mask_url(celery.conf.result_backend),
            'redis_url_file_exists': os.path.exists(os.path.join(SCRIPT_DIR, 'redis_url.txt')),
            'env_file_exists': os.path.exists(os.path.join(SCRIPT_DIR, '.env'))
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

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
    # Content Security Policy (allow Cloudflare, Google Ads, Fonts, data URIs for videos)
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://pagead2.googlesyndication.com https://static.cloudflareinsights.com https://ep2.adtrafficquality.google; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; img-src 'self' data: https:; media-src 'self' data:; connect-src 'self' https:; frame-src https://pagead2.googlesyndication.com https://googleads.g.doubleclick.net https://tpc.googlesyndication.com https://ep2.adtrafficquality.google https://www.google.com;"
    # Remove server header
    response.headers.pop('Server', None)
    return response

# ============================================================================
# Configuration - ALL ON D DRIVE
# ============================================================================

# Security - Generate secret key for session encryption
SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['SECRET_KEY'] = SECRET_KEY

# Redis configuration (for queue + caching) - NO LOCALHOST FALLBACK
# Priority: 1) redis_url.txt (local dev), 2) REDIS_URL env var (Railway)
REDIS_URL = None
# SCRIPT_DIR is already defined above as os.path.dirname(os.path.abspath(__file__))
redis_url_file = os.path.join(SCRIPT_DIR, 'redis_url.txt')
if os.path.exists(redis_url_file):
    with open(redis_url_file, 'r') as f:
        REDIS_URL = f.read().strip()
    print(f"[OK] Auto-loaded Redis URL from {redis_url_file}")
    print(f"   Using: {REDIS_URL}")
else:
    # Fallback to environment variable (Railway deployment)
    REDIS_URL = os.getenv('REDIS_URL')
    if REDIS_URL:
        print(f"[OK] Using REDIS_URL from environment")
        print(f"   Using: {REDIS_URL}")
    else:
        print(f"[ERROR] No Redis URL found!")
        print(f"   - redis_url.txt not found at {redis_url_file}")
        print(f"   - REDIS_URL environment variable not set")
        print(f"   Redis connection will fail - set REDIS_URL in Railway or create redis_url.txt")

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

def get_tunnel_url():
    """Get tunnel URL from environment variable or file"""
    # Check environment variable first
    env_url = os.getenv('TUNNEL_URL')
    if env_url:
        return env_url.strip()

    # Check tunnel_output.txt (localtunnel format: "your url is: https://...")
    tunnel_file = os.path.join(SCRIPT_DIR, 'tunnel_output.txt')
    if os.path.exists(tunnel_file):
        try:
            with open(tunnel_file, 'r') as f:
                for line in f:
                    if 'your url is:' in line:
                        url = line.split('your url is:')[1].strip()
                        if url:
                            return url
        except Exception:
            pass

    # Check web/tunnel_url.txt (alternative format)
    tunnel_file = os.path.join(SCRIPT_DIR, 'web', 'tunnel_url.txt')
    if os.path.exists(tunnel_file):
        try:
            with open(tunnel_file, 'r') as f:
                url = f.read().strip()
                if url:
                    return url
        except Exception:
            pass

    return None

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
            # Removed torch.cuda.synchronize() - was creating hard barrier between parallel segments
            print(f"[CLEANUP] GPU memory cleared: {torch.cuda.memory_allocated() / 1024**2:.1f}MB allocated")
    except Exception as e:
        print(f"[WARNING]  GPU memory clear failed: {e}")
        pass

def performance_checkpoint(stage_name, start_time=None):
    """
    Performance checkpoint logging for identifying bottlenecks.
    """
    import time
    current_time = time.perf_counter()
    
    if start_time is not None:
        elapsed = current_time - start_time
        print(f"[TIME]  {stage_name} completed in {elapsed:.2f} seconds")
    else:
        print(f"[RUNNING] {stage_name} started")
    
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
    print(f"\n[SEGMENT] Processing segment {seg_label}: frames {start_frame}-{end_frame} ({seg_duration} frames)")

    crop_x, crop_y, crop_w, crop_h = calculate_crop_region(seg_bbox, width, height, padding_ratio=0.2, min_size=128)
    speedup = estimate_speedup((width, height), (crop_w, crop_h))
    print(f"   [CROP] Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h} (estimated {speedup:.1f}x speedup)")

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
                print(f"[WARNING]  Warning: Frame {frame_idx} ({frame_idx:04d}.png) not found, skipping")
                continue
            shutil.copy2(src, dst)
            frames_copied += 1

        if frames_copied == 0:
            raise RuntimeError(f"No frames copied for segment {seg_idx}")

        print(f"   [CROP]  Cropping {frames_copied} frames to {crop_w}x{crop_h}...")
        for frame_idx in range(frames_copied):
            frame_file = f"{frame_idx:04d}.png"
            frame_path = os.path.join(seg_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[WARNING]  Warning: Could not read frame {frame_file}, skipping crop")
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

        print(f"   [PAINT] Running faster-propainter pipeline on cropped segment...")
        try:
            # Use cached ProPainter pipeline (pre-loaded at worker startup)
            faster_propainter_pipeline = get_propainter_pipeline()

            import torch
            use_fp16 = torch.cuda.is_available()

            print(f"   [RUNNING] Direct pipeline: segment {seg_idx+1}, resolution={crop_w}x{crop_h}, neighbor_length=10, ref_stride=10, subvideo_length=120, raft_iter=10, FP16={use_fp16}")

            # Each process gets its own CUDA context for true parallel processing
            faster_propainter_pipeline(
                video=seg_cropped_dir,
                mask=seg_mask_dir,
                output=seg_output_dir,
                resize_ratio=1.0,
                mask_dilation=4,
                ref_stride=10,
                neighbor_length=10,
                subvideo_length=120,
                raft_iter=10,
                mode="video_inpainting",
                save_frames=True,
                fp16=use_fp16
            )

            print(f"   [OK] faster-propainter segment {seg_idx+1} completed")

        except Exception as exc:
            print(f"[ERROR] faster-propainter failed on segment {seg_idx}: {exc}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"faster-propainter failed on segment {seg_idx}: {exc}") from exc

        seg_propainter_frames = os.path.join(seg_output_dir, os.path.basename(seg_cropped_dir), 'frames')
        if not os.path.exists(seg_propainter_frames):
            raise RuntimeError(f"ProPainter output frames not found for segment {seg_idx}")

        print(f"   🔗 Merging cleaned region back to full frames (GPU-accelerated)...")

        # Try GPU-accelerated merge first, fallback to CPU if needed
        try:
            import torch
            if torch.cuda.is_available():
                # GPU-accelerated batch processing
                print(f"   [RUNNING] Using GPU for frame merging...")
                for frame_idx in range(seg_duration):
                    frame_file = f"{frame_idx:04d}.png"
                    orig = cv2.imread(os.path.join(seg_frames_dir, frame_file))
                    clean = cv2.imread(os.path.join(seg_propainter_frames, frame_file))

                    if clean is not None and orig is not None:
                        # Convert to GPU tensors (BGR to RGB not needed for merging)
                        orig_gpu = torch.from_numpy(orig).cuda()
                        clean_gpu = torch.from_numpy(clean).cuda()

                        # Merge on GPU
                        orig_gpu[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = clean_gpu

                        # Copy back to CPU and save
                        result_frame = orig_gpu.cpu().numpy()
                        cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), result_frame)
                    elif orig is not None:
                        cv2.imwrite(os.path.join(seg_cleaned_dir, frame_file), orig)

                print(f"   [OK] GPU merge completed")
            else:
                raise RuntimeError("CUDA not available")
        except Exception as e:
            # Fallback to CPU merge
            print(f"   [WARNING]  GPU merge failed ({e}), falling back to CPU...")
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
    Encode a segment's cleaned frames to video (GPU-accelerated with NVENC).

    Returns a tuple (seg_idx, seg_video_path).
    """
    import subprocess

    base_name = context['base_name']
    unique_suffix = context['unique_suffix']
    fps = context['fps']

    seg_prefix = f"{base_name}_{unique_suffix}_seg{seg_idx}"
    seg_video_path = os.path.join(TEMP_DIR, f"{seg_prefix}.mp4")

    seg_label = f"{seg_idx + 1}/{total_segments}"
    print(f"   [ENCODE]  Encoding segment {seg_label} to video (GPU NVENC)...")

    try:
        # Try faster preset p1 for maximum speed, fallback to p4 if fails
        encode_cmd = [
            FFMPEG_EXE, '-y',
            '-framerate', str(fps),
            '-i', os.path.join(seg_cleaned_dir, '%04d.png'),
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',  # Fastest NVENC preset (was p4)
            '-b:v', '8M',  # Increased bitrate for better quality at higher speed
            '-bufsize', '16M',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            seg_video_path
        ]
        result = subprocess.run(encode_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback to p4 if p1 fails
            print(f"   [WARNING]  p1 preset failed, trying p4...")
            encode_cmd[encode_cmd.index('-preset') + 1] = 'p4'
            subprocess.run(encode_cmd, capture_output=True, check=True)

        print(f"   [OK] Segment {seg_label} encoded successfully")
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
        print(f"[WARNING]  Failed to save detection debug image: {exc}")
        return None

# Initialize Celery - NO LOCALHOST FALLBACK
# Priority: CELERY_BROKER_URL env var -> REDIS_URL env var -> redis_url.txt (already loaded into REDIS_URL)
broker = os.getenv("CELERY_BROKER_URL") or os.getenv("REDIS_URL") or REDIS_URL
backend = os.getenv("CELERY_RESULT_BACKEND") or broker

if not broker:
    print("[ERROR] No Celery broker URL configured!")
    print("   Set CELERY_BROKER_URL or REDIS_URL environment variable")
    raise ValueError("Celery broker URL is required - no localhost fallback allowed")

print(f"[DEBUG] Celery Broker = {broker}")
print(f"[DEBUG] Celery Backend = {backend}")

celery = Celery(app.name, broker=broker, backend=backend)
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
    worker_prefetch_multiplier=4,  # Prefetch up to 4 tasks for TRUE parallel execution
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

# ============================================================================
# BACKGROUND ENCODER THREAD (Real-Time Continuous Encoding)
# ============================================================================
# Workers signal when segments ready → Background thread encodes immediately
# Encoding happens continuously as segments complete (not all at once at end)
# Everything stays on ONE GPU for extreme speed

import threading

encoder_thread = None

def start_background_encoder(**kwargs):
    """
    Start background encoding thread when worker process initializes.
    This runs ONCE per worker process.
    """
    global encoder_thread

    print("[BACKGROUND ENCODER INIT] Worker process starting, initializing background encoder...")

    if encoder_thread is None or not encoder_thread.is_alive():
        print("[BACKGROUND ENCODER] Starting real-time encoding thread...")
        encoder_thread = threading.Thread(
            target=background_encoder_worker,
            daemon=True,  # Dies with worker process
            name="BackgroundEncoder"
        )
        encoder_thread.start()
        print("[BACKGROUND ENCODER] Thread active - will encode segments as they complete!")
    else:
        print("[BACKGROUND ENCODER] Thread already running")


def background_encoder_worker():
    """
    Background thread that encodes segments in real-time as they complete.
    Uses Redis pub/sub to receive segment completion notifications.
    PARALLEL ENCODING: Encodes multiple segments simultaneously using ThreadPoolExecutor.
    """
    import json
    import traceback
    import time
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

            pubsub.subscribe('segment_ready')

            print("[BACKGROUND ENCODER] Listening for segment completion signals...")
            print("[BACKGROUND ENCODER] Socket keepalive enabled - NO timeout!")
            print("[BACKGROUND ENCODER] PARALLEL MODE: Up to 4 concurrent NVENC streams!")

            # Create thread pool for parallel encoding (4 matches SEGMENT_WORKERS)
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="EncoderThread") as executor:
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

                            print(f"[BACKGROUND ENCODER] Segment {seg_idx+1}/{total_segments} ready for video {video_id} - submitting to parallel encoder!")

                            # Submit encoding to thread pool (NON-BLOCKING!)
                            future = executor.submit(encode_segment_background, redis_client, data)
                            video_futures[video_id][seg_idx] = future

                            # Check if all segments have been submitted for this video
                            if len(video_futures[video_id]) == total_segments:
                                print(f"[BACKGROUND ENCODER] All {total_segments} segments submitted for {video_id} - waiting for completion...")

                                # Wait for all encoding futures to complete
                                completed_count = 0
                                failed_segments = []

                                for seg_idx, future in video_futures[video_id].items():
                                    try:
                                        future.result()  # Block until this segment completes
                                        completed_count += 1
                                        print(f"[BACKGROUND ENCODER] ✓ Segment {seg_idx+1} encoded! Progress: {completed_count}/{total_segments}")
                                    except Exception as e:
                                        print(f"[BACKGROUND ENCODER ERROR] Segment {seg_idx+1} failed: {e}")
                                        traceback.print_exc()
                                        failed_segments.append(seg_idx)

                                # Finalize if all segments succeeded
                                if not failed_segments:
                                    print(f"[BACKGROUND ENCODER] All {total_segments} segments encoded for {video_id}! Triggering finalization...")
                                    trigger_finalization(redis_client, video_id, total_segments)
                                    print(f"[BACKGROUND ENCODER] ✓ Video {video_id} finalized!")
                                else:
                                    print(f"[BACKGROUND ENCODER ERROR] Video {video_id} had {len(failed_segments)} failed segments: {failed_segments}")

                                # Cleanup tracking
                                del video_futures[video_id]
                                del video_metadata[video_id]

                        except Exception as e:
                            print(f"[BACKGROUND ENCODER ERROR] Failed to process segment: {e}")
                            traceback.print_exc()

        except Exception as e:
            print(f"[BACKGROUND ENCODER] Connection lost: {e}")
            traceback.print_exc()
            print("[BACKGROUND ENCODER] Reconnecting in 2 seconds...")
            time.sleep(2)
            # Loop continues - auto-reconnect!


def encode_segment_background(redis_client, data):
    """
    Encode a segment using GPU NVENC (same encoding as before, just in background).
    Called by background encoder thread continuously as segments complete.
    """
    import subprocess
    import time

    video_id = data['video_id']
    seg_idx = data['seg_idx']
    total_segments = data['total_segments']

    # Get segment metadata from Redis
    segment_key = f"video:{video_id}:segment:{seg_idx}"
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
    # 🔥 SHARED BUFFER: Get frame range for this segment
    start_frame = int(segment_info.get('start_frame', 0))
    end_frame = int(segment_info.get('end_frame', frame_count - 1))

    if not cleaned_dir or not os.path.exists(cleaned_dir):
        raise RuntimeError(f"Cleaned frames directory not found: {cleaned_dir}")

    # Create output path
    seg_video_path = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_seg{seg_idx}.mp4")

    print(f"[ENCODER] Encoding segment {seg_idx}: frames {start_frame}-{end_frame} ({frame_count} frames) @ {fps} fps...")
    encode_start = time.time()

    # 🔥 SHARED BUFFER: Create file list for this segment's frames only
    # (frames are named by global index in shared buffer)
    file_list_path = os.path.join(TEMP_DIR, f"encode_seg{seg_idx}_{video_id}.txt")
    with open(file_list_path, 'w') as f:
        for global_idx in range(start_frame, end_frame + 1):
            frame_path = os.path.join(cleaned_dir, f"{global_idx:04d}.png")
            if os.path.exists(frame_path):
                # Write absolute path for ffmpeg concat
                abs_path = os.path.abspath(frame_path).replace('\\', '/')
                # Use duration 1/fps for each frame
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {1/fps}\n")
        # Last frame needs to be repeated for proper duration
        last_frame_path = os.path.join(cleaned_dir, f"{end_frame:04d}.png")
        if os.path.exists(last_frame_path):
            abs_path = os.path.abspath(last_frame_path).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    # Encode with NVENC using file list (concat demuxer)
    encode_cmd = [
        FFMPEG_EXE, '-y',
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
        encode_duration = time.time() - encode_start

        encoded_size_mb = os.path.getsize(seg_video_path) / (1024 * 1024)
        fps_actual = frame_count / encode_duration if encode_duration > 0 else 0

        print(f"[ENCODER] [OK] Encoded: {encoded_size_mb:.2f} MB in {encode_duration:.2f}s ({fps_actual:.1f} fps)")

        # Store encoded path in Redis
        redis_client.hset(segment_key, 'encoded_path', seg_video_path)
        redis_client.hset(segment_key, 'status', 'encoded')

        # Cleanup file list and frames after successful encoding
        if os.path.exists(file_list_path):
            os.remove(file_list_path)

        # Note: Don't cleanup shared_cleaned_dir yet - other segments may need it!
        # Final cleanup happens after all segments are encoded

    except subprocess.CalledProcessError as e:
        print(f"[ENCODER ERROR] Encoding failed for segment {seg_idx}!")
        print(f"   stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        print(f"[ENCODER ERROR] Encoding timed out after 300s for segment {seg_idx}")
        raise


def trigger_finalization(redis_client, video_id, total_segments):
    """
    Concatenate all encoded segments and merge audio.
    Called automatically when all segments are encoded.
    """
    import subprocess

    print(f"\n[FINALIZE] Starting finalization for video {video_id}")

    # Collect all segment video paths from Redis (in order)
    segment_paths = []
    for seg_idx in range(total_segments):
        segment_key = f"video:{video_id}:segment:{seg_idx}"
        encoded_path_raw = redis_client.hget(segment_key, 'encoded_path')

        # Decode bytes to string
        encoded_path = encoded_path_raw.decode() if isinstance(encoded_path_raw, bytes) else encoded_path_raw

        if not encoded_path or not os.path.exists(encoded_path):
            raise RuntimeError(f"Missing encoded segment {seg_idx}: {encoded_path}")

        segment_paths.append(encoded_path)

    print(f"[FINALIZE] Found {len(segment_paths)} encoded segments")

    # Get video metadata (decode bytes)
    base_name_raw = redis_client.get(f"video:{video_id}:base_name")
    base_name = base_name_raw.decode() if isinstance(base_name_raw, bytes) else (base_name_raw or 'video')

    video_path_raw = redis_client.get(f"video:{video_id}:video_path")
    video_path = video_path_raw.decode() if isinstance(video_path_raw, bytes) else video_path_raw

    # Create concat list
    concat_list_path = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_concat.txt")
    with open(concat_list_path, 'w') as f:
        for seg_path in segment_paths:
            abs_path = os.path.abspath(seg_path).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    # Concatenate with copy codec (instant - no re-encoding)
    temp_processed = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_processed.mp4")

    concat_cmd = [
        FFMPEG_EXE, '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',  # No re-encoding - instant!
        temp_processed
    ]

    print(f"[FINALIZE] Concatenating segments with copy codec...")
    subprocess.run(concat_cmd, capture_output=True, check=True, text=True, timeout=60)
    concat_size_mb = os.path.getsize(temp_processed) / (1024 * 1024)
    print(f"[FINALIZE] ✓ Concatenated: {concat_size_mb:.2f} MB")

    # Merge audio from original
    final_output = os.path.join(RESULT_DIR, f"{base_name}_propainter.mp4")

    if video_path and os.path.exists(video_path):
        # Check if original has audio
        check_audio_cmd = [
            FFPROBE_EXE, '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        has_audio_check = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
        has_audio = 'audio' in has_audio_check.stdout

        if has_audio:
            print(f"[FINALIZE] Merging audio from original...")
            merge_cmd = [
                FFMPEG_EXE, '-y',
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
            print(f"[FINALIZE] ✓ Audio merged")
        else:
            os.rename(temp_processed, final_output)
            print(f"[FINALIZE] No audio in original")
    else:
        os.rename(temp_processed, final_output)
        print(f"[FINALIZE] Using processed video only")

    # Cleanup
    if os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    for seg_path in segment_paths:
        if os.path.exists(seg_path):
            os.remove(seg_path)

    # 🔥 SHARED BUFFER: Cleanup shared frame directory after finalization
    # Get shared_cleaned_dir from first segment's metadata
    segment_key = f"video:{video_id}:segment:0"
    shared_cleaned_dir_raw = redis_client.hget(segment_key, 'cleaned_dir')
    if shared_cleaned_dir_raw:
        shared_cleaned_dir = shared_cleaned_dir_raw.decode() if isinstance(shared_cleaned_dir_raw, bytes) else shared_cleaned_dir_raw
        if shared_cleaned_dir and os.path.exists(shared_cleaned_dir):
            print(f"[FINALIZE] Cleaning up shared frame buffer: {shared_cleaned_dir}")
            shutil.rmtree(shared_cleaned_dir, ignore_errors=True)

    final_size_mb = os.path.getsize(final_output) / (1024 * 1024)
    print(f"[FINALIZE] ✓ Final video ready: {final_output} ({final_size_mb:.2f} MB)")

    # Store final result in Redis
    redis_client.set(f"video:{video_id}:final_path", final_output)
    redis_client.set(f"video:{video_id}:status", "complete")

# ============================================================================
# REDIS VIDEO DOWNLOAD POLLING (for multi-PC parallel downloads)
# ============================================================================

def start_redis_download_poller():
    """
    Background thread that polls Redis for video download signals.
    When prepare_video sets 'video_download:{video_id}' key, this thread
    detects it and downloads the video to cache immediately.

    This enables ALL workers to download in parallel instead of sitting idle.
    """
    import threading
    import time
    import requests

    def poll_redis():
        print("[POLL] Starting Redis video download poller...")
        redis_client = celery.backend.client
        checked_videos = set()  # Track videos we've already processed

        while True:
            try:
                # Scan for all video_download:* keys
                keys = redis_client.keys('video_download:*')

                for key in keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    video_id = key_str.replace('video_download:', '')

                    # Skip if we've already downloaded this video
                    if video_id in checked_videos:
                        continue

                    # Get download URL from Redis
                    download_url = redis_client.get(key)
                    if not download_url:
                        continue

                    download_url = download_url.decode('utf-8') if isinstance(download_url, bytes) else download_url

                    # Check if we already have this video cached
                    cache_dir = os.path.join(TEMP_DIR, 'video_cache')
                    os.makedirs(cache_dir, exist_ok=True)
                    cached_video = os.path.join(cache_dir, f"{video_id}.mp4")

                    if os.path.exists(cached_video):
                        print(f"[OK] Worker {os.getpid()}: Video {video_id} already cached (skip)")
                        checked_videos.add(video_id)
                        continue

                    # Download video to cache
                    print(f"📥 Worker {os.getpid()}: Detected download signal for {video_id}")
                    print(f"   [DOWNLOAD]  Downloading: {download_url}")

                    try:
                        r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=120)
                        r.raise_for_status()

                        with open(cached_video, 'wb') as f:
                            f.write(r.content)

                        file_size = os.path.getsize(cached_video) / (1024 * 1024)
                        print(f"   [OK] Worker {os.getpid()}: Downloaded and cached {video_id} ({file_size:.1f}MB)")
                        checked_videos.add(video_id)

                    except Exception as download_error:
                        print(f"   [ERROR] Worker {os.getpid()}: Download failed for {video_id}: {download_error}")
                        # Don't add to checked_videos so we can retry

                # Clean up checked_videos if it gets too large (memory leak prevention)
                if len(checked_videos) > 100:
                    checked_videos.clear()

            except Exception as e:
                print(f"[WARNING]  Redis poller error: {e}")

            # Poll every 1 second
            time.sleep(1)

    # Start polling thread
    poller_thread = threading.Thread(target=poll_redis, daemon=True)
    poller_thread.start()
    print("[OK] Redis video download poller started")


# Start poller when Celery worker is ready
from celery.signals import worker_ready

@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    """Called when Celery worker finishes initialization - pre-warm models"""
    print("[INIT] Worker ready - warming up models...")

    # ============================================================================
    # ENVIRONMENT VARIABLE VALIDATION - Critical for performance!
    # ============================================================================
    print("=" * 70)
    print("WORKER ENVIRONMENT VALIDATION")
    print("=" * 70)

    # Check critical performance environment variables
    env_vars_status = {
        'USE_NEUFLOW': os.environ.get('USE_NEUFLOW', '0'),
        'FORCE_TRT_RAFT': os.environ.get('FORCE_TRT_RAFT', '0'),
        'ENABLE_FLASH_ATTENTION': os.environ.get('ENABLE_FLASH_ATTENTION', '0'),
        'ENABLE_FP8_TRANSFORMER': os.environ.get('ENABLE_FP8_TRANSFORMER', '1'),  # Default ON
        'RFCNET_TORCHTRT': os.environ.get('RFCNET_TORCHTRT', '0'),
    }

    print("Critical Performance Variables:")
    for var, value in env_vars_status.items():
        is_enabled = value in ('1', 'true', 'yes', 'on')
        status_icon = "✓" if is_enabled else "✗"
        print(f"  {status_icon} {var:<25} = {value}")

    # Check if NeuFlow model file exists
    neuflow_path = os.path.join(os.getcwd(), 'faster-propainter-main', 'models', 'neuflow_things.onnx')
    neuflow_exists = os.path.exists(neuflow_path)
    print()
    print("Model Files:")
    print(f"  {'✓' if neuflow_exists else '✗'} NeuFlow v2 ONNX: {neuflow_path}")

    # Performance warnings
    print()
    print("Performance Analysis:")
    if env_vars_status['USE_NEUFLOW'] == '1':
        if neuflow_exists:
            print("  ✓ NeuFlow v2 enabled - OPTIMAL PERFORMANCE (10-70x faster optical flow)")
        else:
            print("  ✗ WARNING: USE_NEUFLOW=1 but model file missing!")
            print("    → Download from: https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases")
    else:
        print("  ⚠ Using PyTorch RAFT - SLOW! (4-5s per segment)")
        print("    → Set USE_NEUFLOW=1 for 10-70x speedup")

    if env_vars_status['ENABLE_FLASH_ATTENTION'] != '1':
        print("  ⚠ Flash Attention disabled - missing 3-5x transformer speedup")

    if env_vars_status['ENABLE_FP8_TRANSFORMER'] == '1':
        print("  ✓ FP8 Transformer quantization enabled - RTX 4090 Ada (5-10x speedup!)")
    else:
        print("  ⚠ FP8 Transformer disabled - missing 1.3-1.5x speedup on Linear layers")

    print("=" * 70)
    print()

    # Start background threads
    start_redis_download_poller()
    start_background_encoder()

    # Pre-load YOLO detector (saves 6-7s on first task)
    try:
        get_detector()
        print("[OK] YOLO detector pre-loaded")
    except Exception as e:
        print(f"[WARNING]  Failed to pre-load YOLO: {e}")

    # Pre-load ProPainter pipeline (saves 1-2s per segment)
    try:
        get_propainter_pipeline()
        print("[OK] ProPainter pipeline pre-loaded")
    except Exception as e:
        print(f"[WARNING]  Failed to pre-load ProPainter: {e}")

    print("[RUNNING] Worker fully initialized and ready!")


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
                        print(f"[CLEANUP]  Cleaned up old file: {filename} (age: {file_age/60:.1f} min)")
        except Exception as e:
            print(f"[WARNING]  Cleanup error in {directory}: {e}")

# Schedule cleanup to run every 10 minutes
import threading
def schedule_cleanup():
    cleanup_old_files()
    threading.Timer(600, schedule_cleanup).start()  # Run every 10 minutes

# Start cleanup scheduler
threading.Thread(target=schedule_cleanup, daemon=True).start()
print("[CLEANUP]  File cleanup scheduler started (runs every 10 minutes)")

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
        os.path.join(SCRIPT_DIR, 'faster-propainter-main', 'watermark.py'),
        os.path.join(SCRIPT_DIR, 'weights', 'ProPainter.pth'),
        os.path.join(SCRIPT_DIR, 'weights', 'raft-things.pth'),
        os.path.join(SCRIPT_DIR, 'weights', 'recurrent_flow_completion.pth'),
    ]

    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        print("[ERROR] ProPainter assets missing:")
        for path in missing:
            print(f"   - {path}")
        print("   Download weights or copy them into the paths above.")
        propainter_ready = False
        return False

    # Check TensorRT DCNv4 files if TensorRT mode is enabled
    if os.getenv('FORCE_TRT_RFCNET', '0') == '1':
        trt_paths = [
            os.path.join(SCRIPT_DIR, 'engines', 'rfcnet', 'rfcnet_dcnv4_fp16.engine'),
            os.path.join(SCRIPT_DIR, 'dcnv4_tensorrt_plugin', 'build', 'Release', 'dcnv4_plugin.dll'),
        ]
        trt_missing = [path for path in trt_paths if not os.path.exists(path)]
        if trt_missing:
            print("[ERROR] FORCE_TRT_RFCNET=1 but TensorRT DCNv4 files missing:")
            for path in trt_missing:
                print(f"   - {path}")
            print("   Build the engine with: python build_rfcnet_trt_engine.py --fp16")
            print("   Or set FORCE_TRT_RFCNET=0 to use PyTorch fallback")
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
        import numpy as np
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

    # Check ProPainter assets (only matters on cloud workers)
    _check_propainter_assets()
    return detector


# Global ProPainter pipeline cache
_propainter_pipeline = None

def get_propainter_pipeline():
    """
    Cached ProPainter pipeline loader.
    Loads the pipeline once and reuses it across all tasks in the same worker.
    """
    global _propainter_pipeline

    if _propainter_pipeline is None:
        print("=" * 60)
        print("Loading ProPainter pipeline...")
        print("=" * 60)
        faster_propainter_path = os.path.join(SCRIPT_DIR, 'faster-propainter-main')
        if faster_propainter_path not in sys.path:
            sys.path.insert(0, faster_propainter_path)

        from watermark import pipeline as faster_propainter_pipeline
        _propainter_pipeline = faster_propainter_pipeline
        print("=" * 60)
        print("[OK] ProPainter pipeline loaded!")
        print("=" * 60)

        # Warmup SageAttention INT8 kernels (compiles once per worker)
        if os.getenv("ENABLE_SAGE_ATTENTION", "1") == "1":
            try:
                from watermark import warmup_sageattention
                import torch
                if torch.cuda.is_available():
                    # Shape: (B, T, H, W, num_heads, head_dim)
                    # Typical ProPainter transformer: batch=1, temporal=16, spatial=64x64, heads=8, dim=64
                    warmup_sageattention(device='cuda', q_shape=(1, 16, 64, 64, 8, 64))
            except Exception as e:
                print(f"[WARNING] SageAttention warmup failed: {e}")

        # Note: TensorRT RFC Net context will be created on first use (lazy init)
        # Expected performance: 9-12ms/frame after first segment warm-up
        if os.getenv('FORCE_TRT_RFCNET', '0') == '1':
            print("[INFO] TensorRT RFC Net enabled (FORCE_TRT_RFCNET=1)")
            print("[INFO] Context will be created on first inference (expect ~500ms warm-up)")
            print("[INFO] Performance after warm-up: ~9-12ms/frame (consistent)")

    return _propainter_pipeline


# ============================================================================
# Celery Tasks (Background Processing)
# ============================================================================

# ============================================================================
# DISTRIBUTED VIDEO PROCESSING TASKS
# These tasks enable multiple workers across different machines to collaborate
# on processing one video together by distributing segments across all GPUs
# ============================================================================

@celery.task(bind=True, name='watermark.broadcast_download')
def broadcast_video_download(self, video_id, video_url, upload_filename):
    """
    Broadcast task: All workers download video simultaneously
    This runs on ALL workers in parallel to pre-cache the video
    """
    try:
        print(f"\n📡 Broadcast download for video {video_id}")

        # Create cache directory
        cache_dir = os.path.join(TEMP_DIR, 'video_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cached_video = os.path.join(cache_dir, f"{video_id}.mp4")

        # Skip if already cached
        if os.path.exists(cached_video):
            print(f"   [OK] Video already cached")
            return {'status': 'cached', 'path': cached_video}

        # Download video
        print(f"   [DOWNLOAD]  Downloading: {video_url}")
        import requests
        r = requests.get(video_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=120)
        r.raise_for_status()

        # Save to cache
        with open(cached_video, 'wb') as f:
            f.write(r.content)

        file_size = os.path.getsize(cached_video) / (1024 * 1024)
        print(f"   [OK] Video downloaded and cached ({file_size:.2f} MB)")

        return {'status': 'downloaded', 'path': cached_video, 'size_mb': file_size}

    except Exception as e:
        print(f"   [ERROR] Broadcast download failed: {e}")
        return {'status': 'failed', 'error': str(e)}


@celery.task(bind=True, name='watermark.prepare_video')
def prepare_video_task(self, video_path, api_base=None, temp_base=None, video_id=None):
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

        # Download video if on remote worker OR if not found locally
        # First extract the local filename (whether video_path is URL or local path)
        from pathlib import PureWindowsPath
        base_name = PureWindowsPath(video_path).name if '\\' in video_path else os.path.basename(video_path)
        local_video_path = os.path.join(UPLOAD_DIR, base_name)

        # Smart check: If file exists locally (same PC workers share uploads/), skip download!
        if os.path.exists(local_video_path):
            print(f"[OK] Video already exists locally: {local_video_path} (skip download)")
            video_path = local_video_path
        elif not os.path.exists(video_path):
            # File not found locally - download from remote
            tunnel = api_base or os.getenv('TUNNEL_URL')
            if tunnel:
                try:
                    from urllib.parse import urljoin
                    import requests
                    download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')
                    print(f"🌐 Downloading video: {download_url}")
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                    r.raise_for_status()
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    with open(local_video_path, 'wb') as f:
                        f.write(r.content)
                    print(f"[OK] Video downloaded: {local_video_path}")
                    video_path = local_video_path
                except Exception as e:
                    raise Exception(f"Failed to download video: {e}")
            else:
                raise Exception(f"Video not found: {video_path}")

        print(f"📹 Preparing video: {video_path} ({os.path.getsize(video_path) / (1024 * 1024):.2f} MB)")

        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Analyzing video'})

        # NVDEC hardware decoder - NO FALLBACKS! (1.16x faster than CPU)
        use_nvdec = os.getenv("ENABLE_NVDEC", "0") == "1"
        nvdec_loader = None
        cap = None

        if use_nvdec:
            # NVDEC REQUIRED - NO CPU FALLBACK!
            from nvdec_video_loader import NVDECVideoLoader
            nvdec_loader = NVDECVideoLoader(video_path, device_id=0)
            props = nvdec_loader.get_properties()
            fps = int(props['fps'])
            width = props['width']
            height = props['height']
            total_frames = props['total_frames']
            print(f"[OK] NVDEC hardware decoder: {width}x{height} @ {fps} fps ({total_frames} frames)")
        else:
            # CPU decoder (if NVDEC disabled)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {video_path}")

            fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            print(f"[OK] CPU decoder: {width}x{height} @ {fps} fps ({total_frames} frames)")

        base_name = Path(video_path).stem
        # Use provided video_id (from upload task_id) for cache consistency, fallback to Celery task ID
        video_id = video_id or (self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8])

        # 🔒 DEDUPLICATION: Check if this video is already being prepared by another worker
        try:
            redis_client = celery.backend.client
            lock_key = f'prepare_lock:{base_name}'

            # Check if another worker is already processing this video
            if redis_client.exists(lock_key):
                print(f"[SKIP]  Video '{base_name}' already being prepared by another worker - skipping duplicate task")
                # Clean up video decoder (NO FALLBACKS!)
                if nvdec_loader is not None:
                    nvdec_loader.close()
                if cap is not None:
                    cap.release()
                return {
                    'status': 'skipped',
                    'message': f'Video already being processed',
                    'video_id': base_name
                }

            # Acquire lock for 5 minutes (YOLO detection + frame extraction time)
            redis_client.setex(lock_key, 300, self.request.id if hasattr(self.request, 'id') else 'unknown')
            print(f"🔒 Acquired processing lock for video '{base_name}'")
        except Exception as e:
            print(f"[WARNING]  Deduplication check failed: {e} - proceeding anyway")

        # [RUNNING] REDIS SIGNAL: Tell ALL workers to download this video in parallel!
        # This allows idle workers to start downloading immediately instead of waiting
        try:
            redis_client = celery.backend.client
            # Construct download URL for workers to use
            tunnel = api_base or temp_base or os.getenv('TUNNEL_URL') or os.getenv('API_BASE_URL')
            if tunnel:
                from urllib.parse import urljoin
                video_filename = os.path.basename(video_path)
                download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{video_filename}')
                # Set Redis key with 5 minute expiry (workers will poll for this)
                redis_client.setex(f'video_download:{video_id}', 300, download_url)
                print(f"📡 Redis signal sent: video_download:{video_id} = {download_url}")
        except Exception as e:
            print(f"[WARNING]  Failed to send Redis download signal: {e}")

        # Create shared directories for distributed access
        shared_mask_dir = os.path.join(PROPAINTER_MASK_ROOT, f"{base_name}_{video_id}")
        shared_frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_originals")
        os.makedirs(shared_mask_dir, exist_ok=True)
        os.makedirs(shared_frames_dir, exist_ok=True)

        # [RUNNING] CACHE CHECK: Skip YOLO if segments already cached from previous request
        segments = None
        frames_processed = total_frames
        cache_key = f'segments_cache:{base_name}'

        try:
            # Check if segments were already detected for this video
            cached_data = redis_client.get(cache_key)
            if cached_data:
                import json
                import time
                cached = json.loads(cached_data)

                # Verify frames/masks still exist (not cleaned up)
                frame_count = len([f for f in os.listdir(shared_frames_dir) if f.endswith('.png')]) if os.path.exists(shared_frames_dir) else 0

                if frame_count > 0:
                    segments = cached['segments']
                    frames_processed = cached.get('total_frames', total_frames)
                    frames_with_watermark = cached.get('frames_with_watermark', 0)  # Load from cache
                    cache_age = time.time() - cached.get('cached_at', 0)
                    print(f"[OK] Using cached YOLO results for '{base_name}' (age: {cache_age:.1f}s, {frame_count} frames)")
                    print(f"   [SKIP]  Skipping YOLO detection - reusing {len(segments)} cached segments")
                    # Convert cached segments back to tuples (JSON converts tuples to lists)
                    segments = [tuple(seg) for seg in segments]
                else:
                    print(f"[WARNING]  Cache found but frames missing - will re-run YOLO")
                    segments = None
        except Exception as e:
            print(f"[WARNING]  Cache check failed: {e} - will run YOLO")
            segments = None

        # Run YOLO only if not cached
        if segments is None:
            print(f"[REGEN] Running YOLO detection on {total_frames} frames...")
            self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Detecting watermarks'})

            # [INIT] EXTREME SPEED: Load all frames to memory for batch processing
            print(f"📥 Loading {total_frames} frames to memory (batch processing)...")

            import time
            decode_start = time.time()

            if use_nvdec:
                # NVDEC hardware decoder - NO FALLBACK! (1.16x faster)
                all_frames = nvdec_loader.load_all_frames(to_numpy=True, color_format='BGR')
                frames_processed = len(all_frames)
                decode_time = time.time() - decode_start
                print(f"[OK] NVDEC decoded {frames_processed} frames: {decode_time:.3f}s ({decode_time/frames_processed*1000:.2f}ms/frame)")
                nvdec_loader.close()
            else:
                # CPU decoder (only if NVDEC disabled)
                all_frames = []
                all_frames_reserve = [None] * int(total_frames)  # Reserve memory
                frames_loaded = 0

                # Fast frame loading (no prints in loop!)
                while frames_loaded < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    all_frames.append(frame)
                    frames_loaded += 1

                cap.release()
                frames_processed = len(all_frames)
                decode_time = time.time() - decode_start
                print(f"[OK] CPU decoded {frames_processed} frames: {decode_time:.3f}s ({decode_time/frames_processed*1000:.2f}ms/frame)")

            # [RUNNING] BATCH DETECTION (EXTREME SPEED - 1-2ms per frame!)
            print(f"[RUNNING] Running BATCH detection on {frames_processed} frames (EXTREME SPEED!)...")
            self.update_state(state='PROCESSING', meta={'progress': 15, 'status': f'Batch detection (1-2ms/frame)'})

            import time
            batch_start = time.time()
            # Batch detect ALL frames at once! (batch_size=64 - FASTEST per benchmark: 748 fps!)
            all_detections = detector.detect_batch(all_frames, confidence_threshold=0.15, padding=0, batch_size=64)
            batch_duration = time.time() - batch_start
            ms_per_frame = (batch_duration / max(frames_processed, 1)) * 1000
            print(f"[OK] Batch detection complete: {batch_duration:.3f}s ({ms_per_frame:.2f}ms per frame)")

            # [INIT] EXTREME SPEED: Process detections and store in Redis (in-memory!)
            print(f"⚡ Creating masks and storing in Redis (in-memory)...")
            zero_mask = np.zeros((height, width), dtype=np.uint8)
            bboxes_per_frame = []
            frames_with_watermark = 0
            last_valid_bbox = None
            all_masks = []

            # Track bboxes for segmentation
            for i, detections in enumerate(all_detections):
                if detections:
                    frames_with_watermark += 1
                    last_valid_bbox = detections[0]['bbox']
                    bboxes_per_frame.append(last_valid_bbox)
                elif last_valid_bbox:
                    bboxes_per_frame.append(last_valid_bbox)
                else:
                    bboxes_per_frame.append(None)

            # Create all masks with GPU batch processing (10-17x faster than CPU loop!)
            # OR use SAM2 for temporal consistency (BEST QUALITY!)
            use_sam2 = os.getenv('USE_SAM2_TRACKING', '0') == '1'

            if use_sam2 and frames_with_watermark > 0:
                print(f"⚡ Using SAM2-Tiny for temporal mask tracking (44ms/frame)...")
                mask_start = time.time()

                # Import SAM2 tracker
                import sys
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'faster-propainter-main'))
                from sam2_tracker import SAM2Tracker

                # Get first bbox from YOLO detection
                first_bbox = None
                for bbox in bboxes_per_frame:
                    if bbox is not None:
                        first_bbox = bbox
                        break

                if first_bbox is None:
                    print("[WARNING] No bbox found - falling back to GPU masks")
                    if hasattr(detector, 'use_gpu_masks') and detector.use_gpu_masks:
                        all_masks = detector.create_masks_batch_gpu(all_frames, all_detections)
                    else:
                        all_masks = [
                            detector.create_mask(frame, dets) if dets else zero_mask
                            for frame, dets in zip(all_frames, all_detections)
                        ]
                else:
                    # Initialize SAM2-Tiny tracker
                    tracker = SAM2Tracker(device="cuda")

                    # Convert YOLO bbox [x1, y1, x2, y2] format for SAM2
                    bbox_xyxy = [int(first_bbox[0]), int(first_bbox[1]),
                                 int(first_bbox[2]), int(first_bbox[3])]

                    # Track watermark across all frames
                    masks_bool = tracker.track_from_box(
                        video_frames=all_frames,
                        bbox=bbox_xyxy,
                        frame_idx=0,
                        video_path=video_path
                    )

                    # Convert boolean masks to uint8 [0, 255]
                    all_masks = [(mask * 255).astype(np.uint8) for mask in masks_bool]

                    print(f"[OK] SAM2 tracked {len(all_masks)} frames with temporal consistency")

                mask_duration = time.time() - mask_start
                print(f"   SAM2 tracking: {mask_duration:.3f}s ({mask_duration/frames_processed*1000:.2f}ms/frame)")
            else:
                print(f"⚡ Creating {frames_processed} masks (GPU batch)...")
                mask_start = time.time()

                # Compatibility: Check if GPU masks are available (older commits may not have this)
                if hasattr(detector, 'use_gpu_masks') and detector.use_gpu_masks:
                    # GPU batch processing - ALL masks at once!
                    all_masks = detector.create_masks_batch_gpu(all_frames, all_detections)
                else:
                    # Fallback to CPU sequential (if Kornia not available)
                    all_masks = [
                        detector.create_mask(frame, dets) if dets else zero_mask
                        for frame, dets in zip(all_frames, all_detections)
                    ]

                mask_duration = time.time() - mask_start
                print(f"   Mask creation: {mask_duration:.3f}s ({frames_processed/mask_duration:.1f} masks/sec)")

            # [INIT] EXTREME SPEED: Store in global memory (INSTANT!)
            print(f"⚡ Storing {frames_processed} frames/masks in memory (INSTANT!)...")
            mem_start = time.time()

            # Store in global FRAME_CACHE (just storing Python references - INSTANT!)
            cache_key = f"video_data:{base_name}"
            with FRAME_CACHE_LOCK:
                FRAME_CACHE[cache_key] = {
                    'frames': all_frames,         # List of numpy arrays (already in RAM!)
                    'masks': all_masks,            # List of numpy arrays
                    'bboxes': bboxes_per_frame,   # For segmentation
                    'timestamp': time.time(),      # For cleanup
                    'video_id': video_id,
                    'base_name': base_name
                }

            mem_duration = time.time() - mem_start
            mem_ms_per_frame = (mem_duration / max(frames_processed, 1)) * 1000
            print(f"[OK] Memory storage complete: {mem_duration:.4f}s ({mem_ms_per_frame:.3f}ms per frame) - INSTANT!")

            # OPTIONAL: Also write to disk for backup/remote workers
            # Disabled by default for EXTREME SPEED (memory-only)
            if os.getenv('WRITE_FRAMES_TO_DISK', '0') == '1':
                print(f"💾 Also writing to disk (backup for remote workers)...")
                for i in range(frames_processed):
                    mask_path = os.path.join(shared_mask_dir, f"{i:04d}.png")
                    cv2.imwrite(mask_path, all_masks[i])
                    frame_path = os.path.join(shared_frames_dir, f"{i:04d}.png")
                    cv2.imwrite(frame_path, all_frames[i])
            else:
                print(f"⚡ Skipping disk writes (pure memory for EXTREME SPEED!)")

            if frames_processed == 0:
                raise RuntimeError("No frames processed - video may be corrupted")

            print(f"[OK] Detection complete: {frames_processed} frames, {frames_with_watermark} with watermarks")
            print(f"[OK] Frames saved to shared storage: {shared_frames_dir}")

            # Detect segments (by watermark position changes)
            from segment_detector import detect_segments, merge_adjacent_segments
            segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
            if segments:
                segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
                print(f"[INFO] Detected {len(segments)} segments for distributed processing")
                for i, (start, end, bbox) in enumerate(segments):
                    print(f"   Segment {i+1}: frames {start}-{end} ({end-start+1} frames), bbox={bbox}")
            else:
                # No segments detected - treat entire video as one segment
                segments = [(0, frames_processed-1, last_valid_bbox if last_valid_bbox else [0,0,width,height])]
                print("[INFO] No segments detected - processing entire video as one segment")

            # 💾 Cache segment results for future duplicate requests (1 hour TTL)
            try:
                import json
                import time
                cache_data = {
                    'segments': segments,  # Will be converted to list by JSON
                    'total_frames': frames_processed,
                    'frames_with_watermark': frames_with_watermark,  # Include for accurate stats
                    'cached_at': time.time()
                }
                redis_client.setex(cache_key, 3600, json.dumps(cache_data))
                print(f"💾 Cached YOLO results for '{base_name}' (1 hour TTL)")
            except Exception as e:
                print(f"[WARNING]  Failed to cache results: {e}")
        else:
            # Using cached segments - release video decoder (NO FALLBACKS!)
            if nvdec_loader is not None:
                nvdec_loader.close()
            if cap is not None:
                cap.release()
            # frames_with_watermark should be loaded from cache, but add fallback
            if 'frames_with_watermark' not in locals():
                frames_with_watermark = 0  # Fallback if not in cache

        # Optional: force time-based splitting to ensure multi-GPU distribution
        try:
            import math
            min_segments = int(os.getenv('MIN_SEGMENTS', '0'))
            min_chunk_frames = int(os.getenv('MIN_CHUNK_FRAMES', '60'))
        except Exception:
            min_segments = 0
            min_chunk_frames = 60

        # Force-split ONLY when YOLO fails (0-1 segments detected)
        # If YOLO found multiple segments, preserve them - they represent distinct watermark regions
        if min_segments and len(segments) <= 1 and frames_processed >= min_chunk_frames:
            # YOLO failed - split video into time-based chunks for parallel processing
            base_seg = segments[0] if segments else (0, frames_processed-1, last_valid_bbox if last_valid_bbox else [0,0,width,height])
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
            print(f"🪓 Force-split enabled (YOLO fallback): created {len(segments)} time chunks (chunk≈{chunk} frames)")

        # Provide a base URL so OTHER workers can fetch frames/masks from this host
        temp_base_url = temp_base or os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL')
        if temp_base_url:
            print(f"🌐 Shared temp base set for workers: {temp_base_url.rstrip('/')}/temp/")
        else:
            print("[WARNING]  No TEMP_BASE_URL/TUNNEL_URL set; only preparing worker can read local frames.")

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
                'video_path': video_path,  # For background encoder audio merge
                'shared_mask_dir': shared_mask_dir,
                'shared_frames_dir': shared_frames_dir,
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

        print(f"[OK] Video prepared for distributed processing: {len(segments)} segments ready")

        # 🔒 REDIS LOCK: Only ONE worker should dispatch segments (others exit here)
        # When multiple workers do YOLO in parallel, they all find the same segments
        # Use Redis lock to ensure only the first one dispatches
        redis_client = celery.backend.client
        dispatch_lock_key = f"dispatch_lock:{video_id}"

        # Try to acquire lock (set if not exists, 60 second expiry)
        lock_acquired = redis_client.set(dispatch_lock_key, self.request.id, nx=True, ex=60)

        if not lock_acquired:
            # Another worker already dispatching segments - this worker's job is done!
            print(f"🔓 Another worker already dispatching segments - exiting early (YOLO parallelization worked!)")
            return {
                'chord_id': f'distributed_{video_id}',
                'status': 'processing',
                'message': f'Parallel YOLO worker (segments being dispatched by another worker)'
            }

        # We got the lock! This worker will dispatch segments
        print(f"🔒 Lock acquired - this worker will dispatch segments")

        # ⚡ OPTIMIZATION: Use Celery chord to dispatch segments in parallel
        # Chord automatically waits for ALL segments to complete, then triggers finalize
        print(f"[INIT] Creating chord: {len(segments)} segment tasks → finalize callback")
        self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Dispatching {len(segments)} parallel tasks'})

        # Add total_segments to each segment data
        for seg in segment_tasks_data:
            seg['total_segments'] = len(segments)

        # Create task signatures (delayed execution) for each segment
        segment_sigs = [process_segment_task.s(seg_data) for seg_data in segment_tasks_data]

        # Create chord: segments run in parallel, finalize runs when ALL complete
        # The chord returns a finalize task ID that we can track
        workflow = chord(segment_sigs)(finalize_video_task.s(prepare_result=result))

        # Store tracking for status endpoint (distributed_ pattern compatibility)
        tracking_key = f"segments:{video_id}"
        celery.backend.set(f"{tracking_key}:total", len(segments))
        celery.backend.set(f"{tracking_key}:prepare_result", json.dumps(result))

        print(f"[OK] Chord dispatched! Segments will run in parallel across workers")
        print(f"   Finalize callback ID: {workflow.id}")
        print(f"   Finalize will auto-trigger when all {len(segments)} segments complete")

        # 🔓 Release processing lock - we're done with preparation
        try:
            redis_client = celery.backend.client
            lock_key = f'prepare_lock:{base_name}'
            redis_client.delete(lock_key)
            print(f"🔓 Released processing lock for video '{base_name}'")
        except Exception as e:
            print(f"[WARNING]  Failed to release lock: {e}")

        # Return distributed task ID for frontend tracking (status endpoint recognizes this pattern)
        return {
            'chord_id': f'distributed_{video_id}',
            'status': 'processing',
            'message': f'Chord workflow: {len(segments)} segments → finalize callback'
        }

    except Exception as e:
        # 🔓 Release processing lock on error
        try:
            redis_client = celery.backend.client
            lock_key = f'prepare_lock:{base_name}'
            redis_client.delete(lock_key)
            print(f"🔓 Released processing lock for video '{base_name}' (error cleanup)")
        except:
            pass  # Ignore cleanup errors

        print(f"[ERROR] Error preparing video: {e}")
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
        video_path = segment_data['video_path']  # For background encoder audio merge

        print(f"\n[SEGMENT] Worker processing segment {seg_idx+1}/{total_segments}: frames {start_frame}-{end_frame}")
        self.update_state(state='STARTED', meta={'progress': 0, 'status': f'Processing segment {seg_idx+1}'})

        # Import required modules
        import subprocess
        from crop_utils import calculate_crop_region

        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"   [CROP] Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

        seg_duration = end_frame - start_frame + 1

        # Create local temp directories for this segment
        seg_prefix = f"{base_name}_{video_id}_seg{seg_idx}"
        seg_frames_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_frames")
        seg_cropped_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_cropped")
        seg_mask_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_masks")
        seg_output_dir = os.path.join(TEMP_DIR, f"{seg_prefix}_output")

        # 🔥 SHARED FRAME BUFFER FIX: All segments merge onto same directory
        # This allows multiple segments to cooperatively edit the same frames
        shared_cleaned_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_all_frames_cleaned")
        seg_cleaned_dir = shared_cleaned_dir  # Backwards compatibility alias

        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir, seg_cleaned_dir]:
            os.makedirs(path, exist_ok=True)

        # Download frames from shared storage (smart: try local first, then individual frames, then video as fallback)
        origin_base = segment_data.get('temp_base_url') or os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL')
        shared_mask_dir = segment_data.get('shared_mask_dir')
        shared_frames_dir = segment_data.get('shared_frames_dir')

        # Multi-PC optimization: Skip frame sharing, download video directly in parallel
        # Smart detection: Check if shared dirs are locally accessible (same PC = fast file copy)
        is_local_worker = shared_frames_dir and os.path.exists(shared_frames_dir)
        is_multi_pc = not is_local_worker

        import requests
        frames_copied = 0

        if is_multi_pc:
            # Multi-PC mode: Each worker downloads video directly (faster, no tunnel congestion)
            print(f"   📦 Multi-PC mode: downloading video directly for frames {start_frame}-{end_frame}...")
            self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Downloading video'})

            api_base = os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL') or origin_base
            upload_filename = segment_data.get('upload_filename')
            video_url = segment_data.get('video_url')
            if (api_base or video_url) and upload_filename:
                try:
                    # Check cache first to avoid re-downloading for multiple segments
                    cache_dir = os.path.join(TEMP_DIR, 'video_cache')
                    os.makedirs(cache_dir, exist_ok=True)
                    cached_video = os.path.join(cache_dir, f"{video_id}.mp4")

                    if os.path.exists(cached_video):
                        print(f"   [OK] Using cached video (skip download)")
                        local_video = cached_video
                    else:
                        video_url = video_url or f"{api_base.rstrip('/')}/uploads/{upload_filename}"
                        print(f"   [DOWNLOAD]  Downloading: {video_url}")
                        r = requests.get(video_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                        r.raise_for_status()
                        # Save to cache for future segments
                        with open(cached_video, 'wb') as f:
                            f.write(r.content)
                        print(f"   💾 Video cached for future segments")
                        local_video = cached_video

                    cap2 = cv2.VideoCapture(local_video)
                    if not cap2.isOpened():
                        raise RuntimeError("Video open failed")
                    current_frame = 0
                    while True:
                        ret, frame = cap2.read()
                        if not ret:
                            break
                        if current_frame > end_frame:
                            break
                        if current_frame >= start_frame:
                            dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
                            cv2.imwrite(dst, frame)
                            frames_copied += 1
                        current_frame += 1
                    cap2.release()
                    # Don't delete - keep cached for other segments!
                except Exception as e:
                    raise RuntimeError(f"Video download failed: {e}")
            else:
                raise RuntimeError(f"No video URL available for segment {seg_idx}")
        else:
            # Single-PC mode: Try frame sharing first
            print(f"   [DOWNLOAD]  Loading frames {start_frame}-{end_frame}...")
            self.update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Loading frames'})

            # [INIT] EXTREME SPEED: Try FRAME_CACHE first (pure memory, INSTANT!)
            cache_key = f"video_data:{base_name}"
            memory_hits = 0
            segment_frames_memory = []  # Store frames in memory (skip disk!)

            # 🔥 TEMPORAL CONTEXT FIX: Extract with neighbor padding for ProPainter
            # neighbor_length=10 means ±5 frames needed for temporal context
            neighbor_padding = 5
            padded_start = max(0, start_frame - neighbor_padding)
            padded_end = end_frame + neighbor_padding  # Will be recalculated with total_frames from cache

            if cache_key in FRAME_CACHE:
                # INSTANT access to frames in memory!
                print(f"   ⚡ Loading from memory cache WITH NEIGHBOR PADDING (ZERO disk I/O!)...")
                with FRAME_CACHE_LOCK:
                    cached = FRAME_CACHE[cache_key]
                    cached_frames = cached['frames']
                    total_frames = len(cached_frames)  # Update total_frames from cache

                    # Recalculate padded_end with correct total_frames
                    padded_end = min(total_frames - 1, end_frame + neighbor_padding)

                    # Extract frames WITH PADDING for temporal context
                    print(f"   [CONTEXT] Extracting frames {padded_start}-{padded_end} (core: {start_frame}-{end_frame}, padding: ±{neighbor_padding})")
                    for frame_idx in range(padded_start, padded_end + 1):
                        if frame_idx < len(cached_frames):
                            frame = cached_frames[frame_idx]
                            segment_frames_memory.append(frame)
                            frames_copied += 1
                            memory_hits += 1

                if memory_hits > 0:
                    print(f"   [OK] Loaded {memory_hits} frames from memory (including ±{neighbor_padding} neighbor padding for temporal context!)")

            # Fallback: Try other sources if memory cache incomplete
            if frames_copied < (end_frame - start_frame + 1):
                for frame_idx in range(start_frame, end_frame + 1):
                    # Skip if already copied from memory
                    if frame_idx - start_frame < memory_hits:
                        continue

                    frame_file = f"{frame_idx:04d}.png"

                    # Priority 2: Local filesystem (if on same machine as prepare task)
                local_frame = os.path.join(shared_frames_dir, frame_file) if shared_frames_dir else None
                if local_frame and os.path.exists(local_frame):
                    dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
                    shutil.copy2(local_frame, dst)
                    frames_copied += 1
                elif origin_base:
                    # Download individual frame from origin host (serving /temp/)
                    try:
                        frame_url = f"{origin_base.rstrip('/')}/temp/{base_name}_{video_id}_originals/{frame_file}"
                        r = requests.get(frame_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=10)
                        if r.ok:
                            dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
                            with open(dst, 'wb') as f:
                                f.write(r.content)
                            frames_copied += 1
                    except Exception as e:
                        print(f"[WARNING]  Failed to download frame {frame_idx}: {e}")

            if frames_copied == 0:
                # Fallback: fetch original video from API uploads and extract only needed frames
                print(f"   [WARNING]  No frames found in shared storage, falling back to video download...")
                api_base = os.getenv('API_BASE_URL') or os.getenv('TUNNEL_URL') or origin_base
                upload_filename = segment_data.get('upload_filename')
                video_url = segment_data.get('video_url')
                if (api_base or video_url) and upload_filename:
                    try:
                        import requests
                        video_url = video_url or f"{api_base.rstrip('/')}/uploads/{upload_filename}"
                        print(f"   [DOWNLOAD]  Fallback: downloading original video for local extraction: {video_url}")
                        r = requests.get(video_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                        r.raise_for_status()
                        local_video = os.path.join(TEMP_DIR, f"seg_{seg_idx}_{upload_filename}")
                        with open(local_video, 'wb') as f:
                            f.write(r.content)
                        cap2 = cv2.VideoCapture(local_video)
                        if not cap2.isOpened():
                            raise RuntimeError("Fallback video open failed")
                        current_frame = 0
                        while True:
                            ret, frame = cap2.read()
                            if not ret:
                                break
                            if current_frame > end_frame:
                                break
                            if current_frame >= start_frame:
                                dst = os.path.join(seg_frames_dir, f"{frames_copied:04d}.png")
                                cv2.imwrite(dst, frame)
                                frames_copied += 1
                            current_frame += 1
                        cap2.release()
                        os.remove(local_video)
                    except Exception as e:
                        raise RuntimeError(f"Fallback frame extraction failed: {e}")
                else:
                    raise RuntimeError(f"No frames available for segment {seg_idx}")

        if frames_copied == 0:
            raise RuntimeError(f"No frames extracted for segment {seg_idx}")

        print(f"   [OK] Loaded {frames_copied} frames ({start_frame}-{end_frame})")

        # Try to get masks from shared location (same PC) or download/regenerate them (different PC)
        masks_downloaded = False

        # [INIT] EXTREME SPEED: Try FRAME_CACHE first (pure memory, INSTANT!)
        print(f"   [MASKS] Loading masks...")
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Loading masks'})

        # 🔥 TEMPORAL CONTEXT FIX: Load masks WITH PADDING (same range as frames)
        masks_needed = list(range(padded_start, padded_end + 1))
        masks_succeeded = 0
        memory_mask_hits = 0
        segment_masks_memory = []  # Store masks in memory (skip disk!)

        # Priority 1: Memory cache (INSTANT!)
        if cache_key in FRAME_CACHE:
            print(f"   ⚡ Loading masks from memory WITH NEIGHBOR PADDING (ZERO disk I/O!)...")
            with FRAME_CACHE_LOCK:
                cached = FRAME_CACHE[cache_key]
                cached_masks = cached['masks']

                print(f"   [CONTEXT] Extracting masks {padded_start}-{padded_end} (core: {start_frame}-{end_frame}, padding: ±{neighbor_padding})")
                for abs_frame_idx in masks_needed:
                    if abs_frame_idx < len(cached_masks):
                        mask = cached_masks[abs_frame_idx]
                        segment_masks_memory.append(mask)
                        masks_succeeded += 1
                        memory_mask_hits += 1

        # Priority 2: Local filesystem (if memory incomplete)
        if masks_succeeded < len(masks_needed):
            try:
                for abs_frame_idx in masks_needed:
                    # Skip if already got from memory
                    if abs_frame_idx - start_frame < memory_mask_hits:
                        continue

                    if shared_mask_dir:
                        mask_filename = f"{abs_frame_idx:04d}.png"
                        shared_mask_path = os.path.join(shared_mask_dir, mask_filename)

                        if os.path.exists(shared_mask_path):
                            local_idx = abs_frame_idx - start_frame
                            local_mask_path = os.path.join(seg_mask_dir, f"{local_idx:04d}.png")
                            shutil.copy2(shared_mask_path, local_mask_path)
                            masks_succeeded += 1
                        else:
                            # Missing mask - break and regenerate all
                            break
            except Exception as copy_err:
                print(f"   [WARNING]  Mask loading failed: {copy_err} - will regenerate")

        if masks_succeeded == len(masks_needed):
            if memory_mask_hits > 0:
                print(f"   [OK] Loaded {masks_succeeded} masks from memory (ZERO disk I/O!)")
            else:
                print(f"   [OK] Copied {masks_succeeded} masks from local storage (fast!)")
            masks_downloaded = True
        else:
            print(f"   [WARNING]  Only {masks_succeeded}/{len(masks_needed)} masks found - will regenerate")

        # Try local filesystem first (same PC - FAST, direct file copy)
        if not masks_downloaded and shared_mask_dir and os.path.exists(shared_mask_dir):
            print(f"   [MASKS] Fallback: Copying masks from local shared directory...")
            self.update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Copying masks'})

            try:
                masks_needed = list(range(start_frame, end_frame + 1))
                masks_succeeded = 0

                for abs_frame_idx in masks_needed:
                    mask_filename = f"{abs_frame_idx:04d}.png"
                    shared_mask_path = os.path.join(shared_mask_dir, mask_filename)

                    if os.path.exists(shared_mask_path):
                        # Save to local segment mask dir with LOCAL index (0000, 0001, etc.)
                        local_idx = abs_frame_idx - start_frame
                        local_mask_path = os.path.join(seg_mask_dir, f"{local_idx:04d}.png")
                        shutil.copy2(shared_mask_path, local_mask_path)
                        masks_succeeded += 1
                    else:
                        # Missing mask - break and regenerate all
                        break

                if masks_succeeded == len(masks_needed):
                    print(f"   [OK] Copied {masks_succeeded} masks from local storage (fast!)")
                    masks_downloaded = True
                else:
                    print(f"   [WARNING]  Only {masks_succeeded}/{len(masks_needed)} masks found locally - will regenerate")
            except Exception as copy_err:
                print(f"   [WARNING]  Local mask copy failed: {copy_err} - will regenerate")

        # Fallback: HTTP download (different PC - SLOW but necessary for distributed workers)
        elif not masks_downloaded and origin_base and shared_mask_dir:
            print(f"   📥 Downloading masks from remote location...")
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
                    print(f"   [OK] Downloaded {masks_succeeded} masks from remote location")
                    masks_downloaded = True
                else:
                    print(f"   [WARNING]  Only {masks_succeeded}/{len(masks_needed)} masks downloaded - will regenerate")
            except Exception as download_err:
                print(f"   [WARNING]  Mask download failed: {download_err} - will regenerate")

        # Define memory pipeline flag (check if we have all frames/masks in memory)
        using_memory_pipeline = (len(segment_frames_memory) == frames_copied and
                                 len(segment_masks_memory) == frames_copied)

        # Fallback: Regenerate masks if not downloaded
        detector = None
        if not masks_downloaded:
            print(f"   [REGEN] Regenerating masks with YOLO BATCH detection on {frames_copied} frames...")
            self.update_state(state='PROCESSING', meta={'progress': 25, 'status': f'Detecting watermarks'})

            detector = get_detector()
            last_valid_bbox = None
            frames_with_watermark = 0
            det_conf = 0.15

            # [INIT] EXTREME SPEED: Use batch detection (2.19ms/frame vs 14ms/frame!)
            # Also fixes TensorRT shape mismatch by using letterbox padding
            print(f"   [RUNNING] Running BATCH detection (EXTREME SPEED + letterbox padding)...")

            # Get frames for detection (from memory or disk)
            frames_for_detection = []
            if using_memory_pipeline:
                frames_for_detection = segment_frames_memory
            else:
                for frame_idx in range(frames_copied):
                    frame_file = f"{frame_idx:04d}.png"
                    frame_path = os.path.join(seg_frames_dir, frame_file)
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frames_for_detection.append(frame)

            # Batch detect ALL frames at once! (RTX 4090 optimized)
            all_detections = detector.detect_batch(frames_for_detection, confidence_threshold=det_conf, padding=0, batch_size=128)

            # Find last valid bbox
            for detections_list in all_detections:
                if detections_list:
                    frames_with_watermark += 1
                    last_valid_bbox = detections_list[0]['bbox']

            print(f"   [OK] Batch detection complete: {frames_with_watermark}/{frames_copied} frames with watermarks")
        else:
            # Masks were downloaded - need to extract bbox from them
            print(f"   [INFO] Using downloaded masks - extracting bbox info...")
            last_valid_bbox = bbox  # Use bbox from segment_data (from centralized detection)
            frames_with_watermark = frames_copied  # Assume all frames have watermarks (already filtered)

        # Update crop region based on detected bbox (calculate_crop_region handles min sizing)
        if last_valid_bbox:
            # Recalculate crop region with detected bbox (includes min_size=128 expansion)
            crop_x, crop_y, crop_w, crop_h = calculate_crop_region(last_valid_bbox, width, height, padding_ratio=0.2, min_size=128)
            print(f"   [CROP] Detected crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        else:
            print(f"   [INFO]  No watermark detected in this chunk - will skip ProPainter")

        # Crop frames to detected watermark region AND create masks on cropped frames
        # [INIT] OPTIMIZATION: Skip disk-based cropping if we have in-memory frames/masks
        if last_valid_bbox and not using_memory_pipeline:
            # Disk-based pipeline (fallback for remote workers or when memory cache unavailable)
            print(f"   [CROP]  Cropping frames to watermark region (disk-based)...")
            for frame_idx in range(frames_copied):
                frame_file = f"{frame_idx:04d}.png"
                frame_path = os.path.join(seg_frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    cv2.imwrite(os.path.join(seg_cropped_dir, frame_file), cropped)
