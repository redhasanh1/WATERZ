"""
Production Server for Watermark Removal SaaS
- Async queue processing with Celery + Redis
- GPU-optimized YOLO detection + ProPainter inpainting
- Keeps your PC usable while serving customers
- Designed for $1M/month scale
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


try:
    _ensure_cuda_torch()
except Exception as _torch_init_exc:
    print(f"⚠️  Torch/CUDA initialization failed: {_torch_init_exc}")
    print("    Continuing without torch; CPU-only endpoints may still work.")
    sys.modules.pop('torch', None)

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
from url_utils import (
    sanitize_video_url,
    is_potentially_watermarked_url,
    pick_best_video_url,
)

app = Flask(__name__, static_folder='web')
# CORS: allow cross-origin calls to /api/* and accept the ngrok header if present
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
    expose_headers=["Content-Disposition"]
)

# Track the latest queued task per client IP to stop stale status polls
_LATEST_TASK_BY_IP = {}

def _client_ip():
    try:
        fwd = request.headers.get('X-Forwarded-For')
        if fwd:
            return fwd.split(',')[0].strip()
    except Exception:
        pass
    return request.remote_addr or 'unknown'

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
        'detector_loaded': detector is not None,
        'propainter_ready': _check_propainter_assets()
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

# Redis configuration (for queue + caching) - with password protection
REDIS_URL = os.getenv('REDIS_URL', 'redis://:watermarkz_secure_2024@localhost:6379/0')

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
    broker_pool_limit=None,  # Allow each worker its own connection (was 1, caused task pickup blocking)
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
# Stuck Task Recovery System
# ============================================================================

# Track monitored tasks: task_id -> (start_time, last_check_time, check_count)
_MONITORED_TASKS = {}

def force_output_recovery(task_id):
    """
    Force-recover video output from ProPainter if task is stuck.
    Finds the latest ProPainter output and copies it to results.
    """
    try:
        print(f"⚠️  Attempting force output recovery for task {task_id}")

        # Find latest ProPainter output video
        propainter_outputs = []
        for root, dirs, files in os.walk(PROPAINTER_OUTPUT_ROOT):
            for file in files:
                if file.endswith('.mp4'):
                    file_path = os.path.join(root, file)
                    mtime = os.path.getmtime(file_path)
                    propainter_outputs.append((mtime, file_path))

        if not propainter_outputs:
            print(f"❌ No ProPainter outputs found for recovery")
            return None

        # Sort by modification time (newest first)
        propainter_outputs.sort(reverse=True)
        latest_video = propainter_outputs[0][1]

        # Copy to results with recovery marker
        recovery_filename = f"recovered_{task_id[:8]}_{int(time.time())}.mp4"
        recovery_path = os.path.join(RESULT_DIR, recovery_filename)
        shutil.copy2(latest_video, recovery_path)

        print(f"✅ Force recovery successful: {recovery_path}")
        print(f"   Source: {latest_video}")

        return recovery_path

    except Exception as e:
        print(f"❌ Force recovery failed for task {task_id}: {e}")
        return None


def monitor_stuck_tasks():
    """
    Monitor tasks for stuck ProPainter processing.
    Runs every 30 seconds and checks for tasks stuck at 'Running ProPainter'.
    """
    from celery.result import AsyncResult

    global _MONITORED_TASKS
    current_time = time.time()

    # Clean up completed/failed tasks from monitoring
    tasks_to_remove = []
    for tid in list(_MONITORED_TASKS.keys()):
        try:
            task = AsyncResult(tid, app=celery)
            if task.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
                tasks_to_remove.append(tid)
        except Exception:
            tasks_to_remove.append(tid)

    for tid in tasks_to_remove:
        del _MONITORED_TASKS[tid]

    # Check all monitored tasks
    for task_id, (start_time, last_check_time, check_count) in list(_MONITORED_TASKS.items()):
        try:
            task = AsyncResult(task_id, app=celery)

            if task.state == 'PROCESSING':
                info = task.info or {}
                status = info.get('status', '')
                progress = info.get('progress', 0)

                # Check if stuck: Running ProPainter for more than 5 minutes
                time_stuck = current_time - start_time

                if 'Running ProPainter' in status and progress < 100 and time_stuck > 300:  # 5 minutes
                    print(f"⚠️  Task {task_id} stuck at '{status}' for {time_stuck/60:.1f} minutes")

                    # Attempt force recovery after 5 minutes
                    recovery_path = force_output_recovery(task_id)

                    if recovery_path:
                        # Update task monitoring
                        _MONITORED_TASKS[task_id] = (start_time, current_time, check_count + 1)
                        print(f"✅ Recovery attempt #{check_count + 1} for task {task_id}")
                    else:
                        print(f"❌ Recovery failed for task {task_id}")
                        _MONITORED_TASKS[task_id] = (start_time, current_time, check_count + 1)
                else:
                    # Update last check time
                    _MONITORED_TASKS[task_id] = (start_time, current_time, check_count)

        except Exception as e:
            print(f"⚠️  Error monitoring task {task_id}: {e}")

    # Schedule next check
    threading.Timer(30, monitor_stuck_tasks).start()


# Start stuck task monitor
threading.Thread(target=monitor_stuck_tasks, daemon=True).start()
print("🔍 Stuck task recovery monitor started (checks every 30 seconds)")

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

    _check_propainter_assets()
    return detector


# ============================================================================
# Celery Tasks (Background Processing)
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
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                    r.raise_for_status()

                    # Verify we got actual image content
                    if len(r.content) < 1000:  # Less than 1KB is suspicious
                        print(f"⚠️  Downloaded file is too small ({len(r.content)} bytes), may be an error page")
                        self.update_state(
                            state='FAILURE',
                            meta={'error': f'Image file not found or was deleted. Please re-upload. (filename: {base_name})'}
                        )
                        raise Exception(f"Image file not found or was deleted: {base_name}")

                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    image_path = remote_cached
                    print(f"✅ Downloaded {len(r.content) / (1024):.2f} KB to: {image_path}")
                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 404:
                        print(f"❌ Image file was deleted from server (404): {base_name}")
                        self.update_state(
                            state='FAILURE',
                            meta={'error': f'Image file not found. It may have been cleaned up. Please re-upload.'}
                        )
                        raise Exception(f"Image file was deleted from server: {base_name}")
                    else:
                        print(f"❌ HTTP error downloading image: {http_err}")
                        self.update_state(
                            state='FAILURE',
                            meta={'error': f'Failed to download image from server: HTTP {http_err.response.status_code}'}
                        )
                        raise Exception(f"Failed to download image: HTTP {http_err.response.status_code}")
                except Exception as dl_err:
                    print(f"❌ Failed to download image from API host: {dl_err}")
                    self.update_state(
                        state='FAILURE',
                        meta={'error': f'Image file not found and remote download failed. Please re-upload.'}
                    )
                    raise Exception(f"Image file not found and remote download failed: {base_name}")
            else:
                print(f"❌ No TUNNEL_URL configured and image not found locally: {image_path}")
                self.update_state(
                    state='FAILURE',
                    meta={'error': 'Image file not found on worker. TUNNEL_URL not configured.'}
                )
                raise Exception(f"Image file not found: {image_path}")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception('Failed to read image')

        height, width = img.shape[:2]
        base_name = Path(image_path).stem
        unique_suffix = self.request.id[:8] if getattr(self.request, 'id', None) else uuid.uuid4().hex[:8]

        self.update_state(state='PROCESSING', meta={'progress': 15, 'status': 'Detecting watermark'})
        detections = det.detect(img, confidence_threshold=0.25, padding=0)

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

        # Generate mask (expand + feather to cover edges and artifacts)
        try:
            pad = int(os.getenv('DETECTOR_PADDING', '20'))
        except Exception:
            pad = 20
        try:
            feather = int(os.getenv('DETECTOR_FEATHER', '21'))
        except Exception:
            feather = 21
        mask = det.create_mask(img, detections, expand_pixels=pad, feather_pixels=feather)
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

        self.update_state(state='PROCESSING', meta={'progress': 45, 'status': 'Running ProPainter with FP16'})

        # Run ProPainter on the frame sequence with FP16 optimization
        cmd = [
            sys.executable,
            PROPAINTER_SCRIPT,
            '-i', frame_dir,
            '-m', mask_dir,
            '-o', PROPAINTER_OUTPUT_ROOT,
            '--save_frames',  # Save individual frames
        ]

        # Default ProPainter tuning for images (balanced fast)
        if '--neighbor_length' not in cmd:
            cmd.extend(['--neighbor_length', '10'])
        if '--ref_stride' not in cmd:
            cmd.extend(['--ref_stride', '20'])
        if '--mask_dilation' not in cmd:
            cmd.extend(['--mask_dilation', '6'])

        # Optional ProPainter tuning via environment variables (applies to image path as well)
        try:
            opt_map = {
                'PROPAINTER_NEIGHBOR_LENGTH': '--neighbor_length',
                'PROPAINTER_REF_STRIDE': '--ref_stride',
                'PROPAINTER_SUBVIDEO_LENGTH': '--subvideo_length',
                'PROPAINTER_RAFT_ITER': '--raft_iter',
                'PROPAINTER_MASK_DILATION': '--mask_dilation',
                'PROPAINTER_WIDTH': '--width',
                'PROPAINTER_HEIGHT': '--height',
            }
            for env_name, flag in opt_map.items():
                val = os.getenv(env_name)
                if val is not None and str(val).strip() != '':
                    ival = int(float(val))
                    cmd.extend([flag, str(ival)])
            # float option for resize ratio
            rr = os.getenv('PROPAINTER_RESIZE_RATIO')
            if rr is not None and str(rr).strip() != '':
                fval = float(rr)
                cmd.extend(['--resize_ratio', str(fval)])
        except Exception as _env_exc:
            print(f"⚠️  Ignoring invalid PROPAINTER_* env: {_env_exc}")

        # Enable FP16 for 2x speedup
        try:
            import torch
            if torch.cuda.is_available():
                cmd.append('--fp16')
                print("✓ ProPainter FP16 enabled for image processing")
        except Exception:
            pass
        if '--fp16' not in cmd and str(os.getenv('PROPAINTER_FP16', '0')).lower() in ('1', 'true', 'yes', 'on'):
            cmd.append('--fp16')

        print(f"Launching ProPainter for image: {' '.join(cmd)}")

        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ ProPainter failed")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("ProPainter inference failed")

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

        return {'path': out_path}
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
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=60)
                    r.raise_for_status()

                    # Verify we got actual video content
                    if len(r.content) < 1000:  # Less than 1KB is suspicious
                        print(f"⚠️  Downloaded file is too small ({len(r.content)} bytes), may be an error page")
                        self.update_state(
                            state='FAILURE',
                            meta={'error': f'Video file not found or was deleted. Please re-upload. (filename: {base_name})'}
                        )
                        raise Exception(f"Video file not found or was deleted: {base_name}")

                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    video_path = remote_cached
                    print(f"✅ Downloaded {len(r.content) / (1024*1024):.2f} MB to: {video_path}")
                except requests.exceptions.HTTPError as http_err:
                    if http_err.response.status_code == 404:
                        print(f"❌ Video file was deleted from server (404): {base_name}")
                        self.update_state(
                            state='FAILURE',
                            meta={'error': f'Video file not found. It may have been cleaned up. Please re-upload.'}
                        )
                        raise Exception(f"Video file was deleted from server: {base_name}")
                    else:
                        print(f"❌ HTTP error downloading video: {http_err}")
                        self.update_state(
                            state='FAILURE',
                            meta={'error': f'Failed to download video from server: HTTP {http_err.response.status_code}'}
                        )
                        raise Exception(f"Failed to download video: HTTP {http_err.response.status_code}")
                except Exception as dl_err:
                    print(f"❌ Failed to download video from API host: {dl_err}")
                    self.update_state(
                        state='FAILURE',
                        meta={'error': f'Video file not found and remote download failed. Please re-upload.'}
                    )
                    raise Exception(f"Video file not found and remote download failed: {base_name}")
            else:
                print(f"❌ No TUNNEL_URL configured and video not found locally: {video_path}")
                self.update_state(
                    state='FAILURE',
                    meta={'error': 'Video file not found on worker. TUNNEL_URL not configured.'}
                )
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
        # Also write frames to disk so we can call ProPainter in 'frames' mode.
        frame_dir = os.path.join(TEMP_DIR, f"video_frames_{base_name}_{unique_suffix}")
        os.makedirs(frame_dir, exist_ok=True)

        # Ultra-speed: optionally process every Nth frame and replicate outputs
        skip_inpaint_every = int(os.getenv('SKIP_INPAINT_EVERY', '2'))
        use_skip = skip_inpaint_every > 1
        if use_skip:
            frame_dir_proc = os.path.join(TEMP_DIR, f"video_frames_reduced_{base_name}_{unique_suffix}")
            mask_dir_proc = os.path.join(TEMP_DIR, f"masks_reduced_{base_name}_{unique_suffix}")
            os.makedirs(frame_dir_proc, exist_ok=True)
            os.makedirs(mask_dir_proc, exist_ok=True)
        else:
            frame_dir_proc = frame_dir
            mask_dir_proc = mask_dir

        zero_mask = np.zeros((height, width), dtype=np.uint8)
        last_valid_bbox = None
        frames_processed = 0
        frames_with_watermark = 0
        reduced_index = 0
        replication_plan = []  # (reduced_idx, start_original_idx, count)
        actual_detections = 0

        # Track bbox per frame for smart cropping segmentation
        bboxes_per_frame = []  # List of bbox tuples or None

        # Detection interval: run YOLO every Nth frame (lower = more frequent)
        detect_interval = int(os.getenv('YOLO_DETECT_INTERVAL', '5'))
        # Detector warmup and sensitivity tuning
        warmup_frames = int(os.getenv('YOLO_DETECT_WARMUP_FRAMES', '60'))
        try:
            detector_conf = float(os.getenv('DETECTOR_CONFIDENCE', '0.25'))
        except Exception:
            detector_conf = 0.25
        hit_found = False

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

            process_this_frame = (frames_processed % max(skip_inpaint_every, 1) == 0) if use_skip else True

            if process_this_frame:
                # Run detector on first warmup frames or at interval
                run_detection = (
                    (not hit_found and frames_processed < warmup_frames) or
                    (frames_processed % max(detect_interval, 1) == 0)
                )
                if run_detection:
                    detections = detector.detect(frame, confidence_threshold=detector_conf, padding=0)
                    active_detections = detections
                    actual_detections += 1

                    if detections:
                        hit_found = True
                        frames_with_watermark += 1
                        primary = detections[0]
                        if primary.get('confidence', 0) >= detector_conf:
                            last_valid_bbox = primary['bbox']
                    elif last_valid_bbox:
                        active_detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]
                else:
                    # Reuse last detected bounding box for frames in between
                    active_detections = (
                        [{'bbox': last_valid_bbox, 'confidence': 0.0}] if last_valid_bbox else []
                    )

                if active_detections:
                    # Expand + feather for robust mask coverage
                    try:
                        pad = int(os.getenv('DETECTOR_PADDING', '20'))
                    except Exception:
                        pad = 20
                    try:
                        feather = int(os.getenv('DETECTOR_FEATHER', '21'))
                    except Exception:
                        feather = 21
                    mask = detector.create_mask(frame, active_detections, expand_pixels=pad, feather_pixels=feather)
                else:
                    mask = zero_mask

                # Save reduced frame/mask with contiguous indices
                frame_path = os.path.join(frame_dir_proc, f"{reduced_index:04d}.png")
                cv2.imwrite(frame_path, frame)
                mask_path = os.path.join(mask_dir_proc, f"{reduced_index:04d}.png")
                cv2.imwrite(mask_path, mask)

                # Plan replication for skipped frames
                remaining_after = max(total_frames - frames_processed, 0)
                replicate_count = min(skip_inpaint_every if use_skip else 1, remaining_after + 1)
                replication_plan.append((reduced_index, frames_processed, replicate_count))
                reduced_index += 1

            # Count overall frames for progress/logs
            frames_processed += 1

        cap.release()

        if frames_processed == 0:
            raise RuntimeError("No frames were processed - video may be corrupted")

        detections_run = actual_detections
        print(f"✅ Masks generated: {frames_processed} frames total")
        print(f"   🎯 YOLO detections: {detections_run}/{frames_processed} frames (detect_interval={detect_interval})")
        print(f"   💧 Frames with watermark: {frames_with_watermark}")

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        self.update_state(
            state='PROCESSING',
            meta={'progress': 55, 'status': 'Running ProPainter'}
        )

        # Prefer frames mode to avoid torchvision video decode issues on some builds
        cmd = [
            sys.executable,
            PROPAINTER_SCRIPT,
            '-i', frame_dir_proc,
            '-m', mask_dir_proc,
            '-o', PROPAINTER_OUTPUT_ROOT,
            '--save_fps', str(fps),
            '--save_frames',
        ]
        # Ultra-speed ProPainter tuning for videos (optimized for Phase 2 cropping)
        if '--neighbor_length' not in cmd:
            cmd.extend(['--neighbor_length', '6'])
        if '--ref_stride' not in cmd:
            cmd.extend(['--ref_stride', '50'])
        if '--raft_iter' not in cmd:
            cmd.extend(['--raft_iter', '5'])
        if '--subvideo_length' not in cmd:
            cmd.extend(['--subvideo_length', '150'])
        if '--mask_dilation' not in cmd:
            cmd.extend(['--mask_dilation', '8'])
        # Process at full resolution (no resize_ratio) for Phase 2 cropped regions
        # Optional ProPainter tuning via environment variables (video path)
        try:
            opt_map = {
                'PROPAINTER_NEIGHBOR_LENGTH': '--neighbor_length',
                'PROPAINTER_REF_STRIDE': '--ref_stride',
                'PROPAINTER_SUBVIDEO_LENGTH': '--subvideo_length',
                'PROPAINTER_RAFT_ITER': '--raft_iter',
                'PROPAINTER_MASK_DILATION': '--mask_dilation',
                'PROPAINTER_WIDTH': '--width',
                'PROPAINTER_HEIGHT': '--height',
            }
            for env_name, flag in opt_map.items():
                val = os.getenv(env_name)
                if val is not None and str(val).strip() != '':
                    ival = int(float(val))
                    cmd.extend([flag, str(ival)])
            rr = os.getenv('PROPAINTER_RESIZE_RATIO')
            if rr is not None and str(rr).strip() != '':
                fval = float(rr)
                cmd.extend(['--resize_ratio', str(fval)])
        except Exception as _env_exc:
            print(f"⚠️  Ignoring invalid PROPAINTER_* env: {_env_exc}")
        try:
            import torch
            if torch.cuda.is_available():
                cmd.append('--fp16')
        except Exception:
            pass
        # Allow forcing FP16 even if torch check fails
        if '--fp16' not in cmd and str(os.getenv('PROPAINTER_FP16', '0')).lower() in ('1', 'true', 'yes', 'on'):
            cmd.append('--fp16')

        # Log summary of actual parameters used
        def _get_flag_val(lst, flag):
            try:
                idx = len(lst) - 1 - lst[::-1].index(flag)
                return lst[idx + 1]
            except ValueError:
                return None
        print(
            "✓ ProPainter ultra: "
            f"{'FP16 + ' if '--fp16' in cmd else ''}"
            f"neighbor_length={_get_flag_val(cmd, '--neighbor_length')} + "
            f"ref_stride={_get_flag_val(cmd, '--ref_stride')} + "
            f"raft_iter={_get_flag_val(cmd, '--raft_iter')} + "
            f"subvideo_length={_get_flag_val(cmd, '--subvideo_length')} + "
            f"resize_ratio={_get_flag_val(cmd, '--resize_ratio')}"
        )
        print(f"Launching ProPainter: {' '.join(cmd)}")

        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ ProPainter failed")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("ProPainter inference failed")

        save_root = os.path.join(PROPAINTER_OUTPUT_ROOT, os.path.basename(frame_dir_proc))
        frames_out = os.path.join(save_root, 'frames')
        if not os.path.isdir(frames_out):
            raise RuntimeError(f"ProPainter frames output not found: {frames_out}")

        temp_processed = os.path.join(RESULT_DIR, f"{base_name}_propainter_video.mp4")

        if use_skip:
            # Reconstruct full-length sequence by replicating reduced outputs
            reconstructed_dir = os.path.join(TEMP_DIR, f"reconstructed_{base_name}_{unique_suffix}")
            os.makedirs(reconstructed_dir, exist_ok=True)

            for red_idx, start_idx, count in replication_plan:
                src = os.path.join(frames_out, f"{red_idx:04d}.png")
                if not os.path.exists(src):
                    raise RuntimeError(f"Missing ProPainter frame: {src}")
                for k in range(count):
                    dest_idx = start_idx + k
                    dest = os.path.join(reconstructed_dir, f"{dest_idx:04d}.png")
                    shutil.copy2(src, dest)

            # Encode reconstructed frames to video
            try:
                import subprocess
                encode_cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(reconstructed_dir, '%04d.png'),
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '18',
                    temp_processed
                ]
                print(f"Encoding reconstructed video: {' '.join(encode_cmd)}")
                enc = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=300)
                if enc.returncode != 0:
                    print(enc.stderr)
                    raise RuntimeError('Failed to encode reconstructed frames')
            except FileNotFoundError:
                raise RuntimeError('ffmpeg not available to encode reconstructed video')
        else:
            # Use ProPainter produced video directly
            produced_video = os.path.join(save_root, 'inpaint_out.mp4')
            if not os.path.exists(produced_video):
                raise RuntimeError(f"ProPainter output not found: {produced_video}")
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
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '18',
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
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '18',
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
            shutil.rmtree(frame_dir, ignore_errors=True)
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

        # Record latest real Celery task id for this client IP
        try:
            ip = _client_ip()
            _LATEST_TASK_BY_IP[ip] = task.id
        except Exception:
            pass

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
        from celery.result import AsyncResult

        # If enabled, require that clients only poll the latest task they created
        require_current = str(os.getenv('STATUS_REQUIRE_CURRENT', '1')).lower() in ('1', 'true', 'yes', 'on')
        if require_current:
            try:
                ip = _client_ip()
                current_for_ip = _LATEST_TASK_BY_IP.get(ip)
                if current_for_ip and current_for_ip != task_id:
                    # Mark as stale and do not log
                    return jsonify({
                        'state': 'STALE',
                        'stale': True,
                        'current_task_id': current_for_ip
                    }), 409
            except Exception:
                pass

        task = AsyncResult(task_id, app=celery)

        # Check if task actually exists in Redis or if it's just an expired/nonexistent task
        # PENDING can mean either "queued and waiting" OR "doesn't exist"
        if task.state == 'PENDING':
            try:
                # Try to get the task result from Redis to see if it actually exists
                redis_client = redis.from_url(REDIS_URL)
                task_key = f"celery-task-meta-{task_id}"
                exists = redis_client.exists(task_key)
                redis_client.close()

                if not exists:
                    # Task doesn't exist in Redis - it either expired or never existed
                    print(f"⚠️  Task {task_id} not found in Redis (expired or invalid)")
                    return jsonify({
                        'state': 'EXPIRED',
                        'error': 'Task not found. It may have expired or been cleaned up. Please submit a new request.'
                    }), 410  # 410 Gone - resource no longer available
            except Exception as redis_err:
                print(f"⚠️  Redis check failed for task {task_id}: {redis_err}")

        # Throttled or suppressed debug logging to avoid flooding from client polls
        suppress = str(os.getenv('STATUS_LOG_SUPPRESS', '0')).lower() in ('1', 'true', 'yes', 'on')
        if not suppress:
            global _STATUS_LOG_LAST
            try:
                STATUS_LOG_THROTTLE_SECONDS = int(os.getenv('STATUS_LOG_THROTTLE_SECONDS', '600'))  # default 10 minutes
            except Exception:
                STATUS_LOG_THROTTLE_SECONDS = 600
            if '_STATUS_LOG_LAST' not in globals():
                _STATUS_LOG_LAST = {}
            now_ts = time.time()
            last_ts = _STATUS_LOG_LAST.get(task_id, 0)
            if STATUS_LOG_THROTTLE_SECONDS <= 0 or (now_ts - last_ts) >= STATUS_LOG_THROTTLE_SECONDS:
                print(f"📊 Status check - Task: {task_id}, State: {task.state}, Info: {task.info}")
                _STATUS_LOG_LAST[task_id] = now_ts

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
            # task.result is a dict with 'path' and 'metadata'
            result_data = task.result
            print(f"🔍 DEBUG - task.result type: {type(result_data)}, value: {result_data}")

            if isinstance(result_data, dict):
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

        # Try Playwright path first; if unavailable in environment, fall back to
        # cookie-authenticated HTTP parsing via requests.
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
            _has_playwright = True
        except Exception:
            _has_playwright = False

        if not _has_playwright:
            print("⚠️  Playwright not available. Falling back to authenticated HTTP parsing.")
            import re as _re
            import html as _html
            import requests as _requests

            # Build a session with Sora cookies
            sess = _requests.Session()
            sess.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })

            try:
                with open(cookies_file, 'r') as f:
                    cookies = json.load(f)
                for c in cookies:
                    # Only add cookies for sora domain (or subdomains)
                    domain = c.get('domain') or 'sora.chatgpt.com'
                    if 'sora.chatgpt.com' in domain:
                        sess.cookies.set(
                            name=c.get('name'),
                            value=c.get('value'),
                            domain=domain,
                            path=c.get('path', '/'),
                        )
            except Exception as cookie_err:
                print(f"❌ Failed to load cookies for HTTP fallback: {cookie_err}")
                return jsonify({'status': 'error', 'message': 'Cookie load failed for fallback'}), 401

            # Fetch page HTML
            try:
                resp = sess.get(url, timeout=30)
                if resp.status_code >= 400:
                    return jsonify({'status': 'error', 'message': f'HTTP {resp.status_code} fetching page'}), 502
                content = resp.text
            except Exception as http_err:
                print(f"❌ HTTP error during fallback: {http_err}")
                return jsonify({'status': 'error', 'message': 'Failed to fetch Sora page'}), 502

            # Extract .mp4 URLs and pick best
            video_urls = _re.findall(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content)
            if not video_urls:
                return jsonify({'status': 'error', 'message': 'No .mp4 URL found in page (fallback)'}), 404

            ordered = [_html.unescape(u) for u in video_urls]
            video_src = pick_best_video_url(ordered) or ordered[0]
            if is_potentially_watermarked_url(video_src):
                cleaned = sanitize_video_url(video_src)
                if cleaned != video_src:
                    print(f"🧼 Watermark flags detected. Using cleaned URL: {cleaned}")
                    video_src = cleaned

            print(f"⬇️  Downloading video via HTTP fallback: {video_src}")
            try:
                with sess.get(video_src, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
            except Exception as dl_err:
                print(f"❌ Fallback download error: {dl_err}")
                return jsonify({'status': 'error', 'message': 'Fallback download failed'}), 500

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Downloaded successfully! Size: {file_size:.2f} MB")

            return jsonify({
                'status': 'success',
                'task_id': task_id,
                'video_url': f'/uploads/{task_id}.mp4'
            })

        # Playwright path
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
                # Prefer a candidate that does not look watermarked
                ordered = [html.unescape(u) for u in video_urls]
                chosen = pick_best_video_url(ordered)
                video_src = chosen or html.unescape(video_urls[0])

                # Final sanitize if watermark toggles detected
                if is_potentially_watermarked_url(video_src):
                    cleaned = sanitize_video_url(video_src)
                    if cleaned != video_src:
                        print(f"🧼 Watermark flags detected. Using cleaned URL: {cleaned}")
                        video_src = cleaned

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
                ordered = [html.unescape(u) for u in video_urls]
                video_src = pick_best_video_url(ordered) or ordered[0]
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

                # Prefer non-watermarked URL if watermark toggles are present
                if is_potentially_watermarked_url(video_src):
                    cleaned = sanitize_video_url(video_src)
                    if cleaned != video_src:
                        print(f"🧼 Watermark flags detected. Using cleaned URL: {cleaned}")
                        video_src = cleaned

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

        # Remember this as the latest client task-id placeholder
        try:
            ip = _client_ip()
            _LATEST_TASK_BY_IP[ip] = task_id  # placeholder until queued
        except Exception:
            pass

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
                result = celery.send_task('watermark.remove_video', args=[video_path])
            print(f"✅ Task queued with ID: {result.id}")

            # Record the latest real Celery task id for this client IP
            try:
                ip = _client_ip()
                _LATEST_TASK_BY_IP[ip] = result.id
            except Exception:
                pass
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
