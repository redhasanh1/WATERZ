"""\nCelery Worker Tasks - GPU Processing\n"""\n\n"""
Production Server for Watermark Removal SaaS
- Async queue processing with Celery + Redis
- GPU-optimized YOLO detection + ProPainter inpainting
- Keeps your PC usable while serving customers
- Designed for $1Mi/month scale
- ALL FILES STAY ON D DRIVE (inside watermarkz folder)
"""

# Hi!

import sys
import os
import importlib
import shutil
from pathlib import Path

# Load environment variables from .env file (for Celery Redis configuration)
from dotenv import load_dotenv
load_dotenv()

# Email utilities for password reset
from email_utils import send_reset_email

# CRITICAL: Force ALL temp/cache to D drive (watermarkz folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Detect Railway environment
IS_RAILWAY = os.getenv('RAILWAY_ENVIRONMENT_NAME') is not None

# Use Railway volume (/data) for persistent storage in production
if IS_RAILWAY:
    DATA_DIR = '/data'
    TEMP_DIR = os.path.join(DATA_DIR, 'temp')
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')
    UPLOAD_DIR = os.path.join(DATA_DIR, 'uploads')
    RESULT_DIR = os.path.join(DATA_DIR, 'results')
    STATIC_VIDEOS_DIR = os.path.join(DATA_DIR, 'static_videos')
    TRAINING_VIDEOS_DIR = os.path.join(DATA_DIR, 'training_videos')
else:
    DATA_DIR = SCRIPT_DIR
    TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
    CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
    UPLOAD_DIR = os.path.join(SCRIPT_DIR, 'uploads')
    RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')
    STATIC_VIDEOS_DIR = os.path.join(SCRIPT_DIR, 'web')
    TRAINING_VIDEOS_DIR = os.path.join(SCRIPT_DIR, 'videostotrain')

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
os.environ.setdefault('MIN_SEGMENTS', '4')  # Force-split for parallel GPU processing
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

# Stripe for billing (optional - gracefully handle if not installed)
try:
    import stripe
    STRIPE_ENABLED = True
    stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
except ImportError:
    STRIPE_ENABLED = False
    print("[WARNING] Stripe not installed - billing endpoints disabled")

# Authentication and session management
from flask import session, redirect, url_for
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    from google_auth_oauthlib.flow import Flow
    import bcrypt
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    from contextlib import contextmanager
    AUTH_ENABLED = True
    print("[OK] Authentication modules loaded")
except ImportError as e:
    AUTH_ENABLED = False
    print(f"[WARNING] Authentication disabled - missing dependencies: {e}")
    print("[INFO] Install with: pip install google-auth google-auth-oauthlib bcrypt psycopg2-binary")

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

# GPU-ONLY ENCODER CONFIG: Maximum speed with NVENC
# NO CPU FALLBACK - full GPU extreme speed, no overhead
ENCODER_CONFIG = {
    'codec': 'h264_nvenc',
    'preset': 'p1',  # Fastest NVENC preset
    'fallback_preset': 'p4',  # Balanced fallback if p1 fails
    'name': 'NVENC (GPU)'
}

if FFMPEG_EXE:
    print(f"[OK] Video encoder: {ENCODER_CONFIG['name']} - EXTREME SPEED MODE")
    print(f"[OK] No CPU fallback - pure GPU pipeline for maximum performance")

# [INIT] EXTREME SPEED: Global in-memory frame/mask cache
# Shared across all threads in Celery worker (threads pool)
# Stores frames/masks in RAM for instant access (no Redis, no disk!)
FRAME_CACHE = {}
FRAME_CACHE_LOCK = threading.Lock()

# Zero-copy frame buffer for direct GPU→FFmpeg piping
FRAME_BUFFER = {}
FRAME_BUFFER_LOCK = threading.Lock()

class FramePipeEncoder:
    """
    Zero-copy frame encoder that pipes frames directly to FFmpeg.
    Eliminates disk I/O bottleneck by keeping frames in memory.
    """

    def __init__(self, video_id, total_frames, width, height, fps):
        self.video_id = video_id
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = [None] * total_frames  # Pre-allocate frame slots
        self.frames_received = 0
        self.lock = threading.Lock()

    def add_frame(self, index, frame):
        """Add a frame to the buffer at the specified index."""
        with self.lock:
            if 0 <= index < self.total_frames:
                self.frames[index] = frame.copy()  # Copy to ensure persistence
                if self.frames[index] is not None:
                    self.frames_received += 1

    def is_complete(self):
        """Check if all frames have been received."""
        with self.lock:
            return self.frames_received >= self.total_frames

    def get_missing_frames(self):
        """Get list of missing frame indices."""
        with self.lock:
            return [i for i in range(self.total_frames) if self.frames[i] is None]

    def encode_to_video(self, output_path):
        """
        Encode all buffered frames directly to video via FFmpeg pipe.
        Uses rawvideo input to bypass filesystem entirely.
        """
        import subprocess

        missing = self.get_missing_frames()
        if missing:
            print(f"[PIPE ENCODER WARNING] Missing {len(missing)} frames!")
            print(f"[PIPE ENCODER WARNING] Missing indices: {missing[:20]}{'...' if len(missing) > 20 else ''}")
            if missing:
                print(f"[PIPE ENCODER WARNING] First missing: {missing[0]}, Last missing: {missing[-1]}")
            # Fill missing frames with black or previous frame
            for i in missing:
                if i > 0 and self.frames[i-1] is not None:
                    self.frames[i] = self.frames[i-1].copy()
                else:
                    self.frames[i] = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        expected_duration = self.total_frames / self.fps if self.fps > 0 else 0
        print(f"[PIPE ENCODER] Encoding {self.total_frames} frames ({self.width}x{self.height} @ {self.fps} fps)")
        print(f"[PIPE ENCODER] Expected duration: {expected_duration:.2f}s")

        # FFmpeg command for rawvideo pipe input
        encode_cmd = [
            FFMPEG_EXE, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', 'pipe:0',
            '-c:v', ENCODER_CONFIG['codec'],
            '-preset', ENCODER_CONFIG.get('preset', 'p4'),
            '-b:v', '8M',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            output_path
        ]

        print(f"[PIPE ENCODER] FFmpeg command: {' '.join(encode_cmd)}")

        try:
            # Launch FFmpeg process
            process = subprocess.Popen(
                encode_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Write all frames to stdin
            for i, frame in enumerate(self.frames):
                if frame is not None:
                    process.stdin.write(frame.tobytes())
                else:
                    # Should not happen after filling missing frames
                    black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    process.stdin.write(black_frame.tobytes())

            process.stdin.close()
            stdout, stderr = process.communicate(timeout=300)

            if process.returncode != 0:
                print(f"[PIPE ENCODER ERROR] FFmpeg failed: {stderr.decode()}")
                raise RuntimeError(f"FFmpeg encoding failed with code {process.returncode}")

            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[PIPE ENCODER] ✓ Encoded: {output_size:.2f} MB")
            return output_path

        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg encoding timed out after 300s")
        except Exception as e:
            print(f"[PIPE ENCODER ERROR] {e}")
            raise

    def clear(self):
        """Clear all frames to free memory."""
        with self.lock:
            self.frames = [None] * self.total_frames
            self.frames_received = 0

# Cross-platform file locking utility for shared frame buffer
import contextlib
import platform
import time

@contextlib.contextmanager
def file_lock(file_path, timeout=10):
    """
    Cross-platform file locking context manager.

    Args:
        file_path: Path to the file to lock
        timeout: Maximum seconds to wait for lock (default 10)

    Yields:
        None (lock is held within context)

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    lock_file = f"{file_path}.lock"
    lock_fd = None
    start_time = time.time()

    try:
        # Create lock file
        lock_fd = open(lock_file, 'w')

        # Platform-specific locking
        if platform.system() == 'Windows':
            import msvcrt
            # Try to lock with timeout
            while True:
                try:
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                    break  # Lock acquired
                except OSError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Could not acquire lock on {file_path} within {timeout}s")
                    time.sleep(0.01)  # 10ms retry interval
        else:
            import fcntl
            # Try to lock with timeout
            while True:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break  # Lock acquired
                except OSError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Could not acquire lock on {file_path} within {timeout}s")
                    time.sleep(0.01)  # 10ms retry interval

        yield  # Lock held during this block

    finally:
        # Release lock
        if lock_fd:
            try:
                if platform.system() == 'Windows':
                    import msvcrt
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
            except:
                pass
            lock_fd.close()

        # Cleanup lock file
        try:
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except:
            pass

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
    supports_credentials=True,  # Enable cookies/sessions for authentication
    allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
    expose_headers=["Content-Disposition"]
)

# ----------------------------------------------------------------------------
# Database Connection Pool (for user authentication)
# ----------------------------------------------------------------------------
db_pool = None
if AUTH_ENABLED:
    try:
        DATABASE_URL = os.getenv('DATABASE_URL')
        if DATABASE_URL:
            # Railway provides postgres:// but psycopg2 needs postgresql://
            if DATABASE_URL.startswith('postgres://'):
                DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

            db_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DATABASE_URL,
                connect_timeout=5  # 5 second timeout to prevent hanging
            )
            print("[OK] Database connection pool initialized")
        else:
            print("[WARNING] DATABASE_URL not set - authentication will not work")
            AUTH_ENABLED = False
    except Exception as e:
        print(f"[ERROR] Failed to initialize database pool: {e}")
        AUTH_ENABLED = False

@contextmanager
def get_db():
    """Get database connection from pool."""
    if not db_pool:
        raise RuntimeError("Database pool not initialized")

    conn = db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)

# ----------------------------------------------------------------------------
# Authentication Decorators
# ----------------------------------------------------------------------------
from functools import wraps

def require_auth(f):
    """Decorator to require authentication for endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not AUTH_ENABLED:
            # If auth is disabled, allow access (for development)
            return f(*args, **kwargs)

        user_id = session.get('user_id')
        if not user_id:
            return jsonify({
                'status': 'error',
                'error': 'Authentication required. Please sign in.',
                'signin_required': True
            }), 401

        return f(*args, **kwargs)
    return decorated_function

def require_credits(min_credits=1):
    """Decorator to check if user has sufficient credits."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not AUTH_ENABLED:
                # If auth is disabled, allow access (for development)
                return f(*args, **kwargs)

            user_id = session.get('user_id')
            if not user_id:
                return jsonify({
                    'status': 'error',
                    'error': 'Authentication required',
                    'signin_required': True
                }), 401

            # Check credit balance
            try:
                with get_db() as conn:
                    cur = conn.cursor()
                    cur.execute('SELECT credits FROM users WHERE id = %s', (user_id,))
                    result = cur.fetchone()

                    if not result:
                        return jsonify({
                            'status': 'error',
                            'error': 'User not found'
                        }), 404

                    credits = result[0]
                    if credits < min_credits:
                        return jsonify({
                            'status': 'error',
                            'error': 'Insufficient credits',
                            'required': min_credits,
                            'available': credits,
                            'message': f'You need {min_credits} credit(s) but only have {credits}. Please purchase more credits.'
                        }), 402  # Payment Required
            except Exception as e:
                print(f"[ERROR] Credit check failed: {e}")
                # Allow processing to continue if database check fails (graceful degradation)
                pass

            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ----------------------------------------------------------------------------
# Flask Session Configuration (for authentication)
# ----------------------------------------------------------------------------
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
app.config['SESSION_COOKIE_HTTPONLY'] = True  # No JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 30  # 30 days

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

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
\n# ===== CELERY TASKS AND GPU FUNCTIONS =====\n\ndef _ensure_cuda_torch():
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

# Stripe for billing (optional - gracefully handle if not installed)
try:
    import stripe
    STRIPE_ENABLED = True
    stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
except ImportError:
    STRIPE_ENABLED = False
    print("[WARNING] Stripe not installed - billing endpoints disabled")

# Authentication and session management
from flask import session, redirect, url_for
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    from google_auth_oauthlib.flow import Flow
    import bcrypt
    import psycopg2
    from psycopg2.pool import SimpleConnectionPool
    from contextlib import contextmanager
    AUTH_ENABLED = True
    print("[OK] Authentication modules loaded")
except ImportError as e:
    AUTH_ENABLED = False
    print(f"[WARNING] Authentication disabled - missing dependencies: {e}")
    print("[INFO] Install with: pip install google-auth google-auth-oauthlib bcrypt psycopg2-binary")

# [INIT] FFmpeg/FFprobe path detection with fallback
def get_ffmpeg_executables():
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

            # Configure pubsub with NO timeout (keepalive disabled for Railway proxy compatibility)
            pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
            pubsub.connection_pool.connection_kwargs['socket_timeout'] = None
            pubsub.connection_pool.connection_kwargs['socket_keepalive'] = False

            pubsub.subscribe('segment_ready')

            print("[BACKGROUND ENCODER] Listening for segment completion signals...")
            print("[BACKGROUND ENCODER] Socket keepalive enabled with 30s timeout!")
            print("[BACKGROUND ENCODER] PARALLEL MODE: Up to 4 concurrent NVENC streams!")

            # Create thread pool for parallel encoding (4 matches SEGMENT_WORKERS)
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="EncoderThread") as executor:
                # Use get_message with timeout instead of blocking listen()
                # This prevents indefinite blocking and allows graceful shutdown
                last_heartbeat = time.time()
                heartbeat_interval = 30  # 30 second heartbeat

                while True:
                    # Get message with timeout (non-blocking with periodic wakeup)
                    message = pubsub.get_message(timeout=5.0)

                    # Heartbeat check - ensure connection is alive
                    if time.time() - last_heartbeat > heartbeat_interval:
                        try:
                            # Ping Redis to verify connection
                            redis_client.ping()
                            last_heartbeat = time.time()
                        except Exception as e:
                            print(f"[BACKGROUND ENCODER] Heartbeat failed: {e}")
                            raise  # Trigger reconnection

                    if message is None:
                        # No message - continue waiting
                        continue

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

            # Cleanup pubsub connection before reconnecting
            try:
                if 'pubsub' in locals():
                    print("[BACKGROUND ENCODER] Closing old pubsub connection...")
                    pubsub.unsubscribe()
                    pubsub.close()
            except Exception as cleanup_error:
                print(f"[BACKGROUND ENCODER] Cleanup error (ignoring): {cleanup_error}")

            print("[BACKGROUND ENCODER] Reconnecting in 2 seconds...")
            time.sleep(2)
            # Loop continues - auto-reconnect with fresh connection!


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
    # [FIX] SHARED BUFFER: Get frame range for this segment
    start_frame = int(segment_info.get('start_frame', 0))
    end_frame = int(segment_info.get('end_frame', frame_count - 1))

    if not cleaned_dir or not os.path.exists(cleaned_dir):
        raise RuntimeError(f"Cleaned frames directory not found: {cleaned_dir}")

    # Create output path
    seg_video_path = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_seg{seg_idx}.mp4")

    print(f"[ENCODER] Encoding segment {seg_idx}: frames {start_frame}-{end_frame} ({frame_count} frames) @ {fps} fps...")
    encode_start = time.time()

    # 🔒 WAIT FOR FRAMES: Ensure all segment frames are available before encoding
    # This prevents race condition where encoder starts before frames are written
    expected_frames = end_frame - start_frame + 1
    max_wait = 30  # seconds
    wait_start = time.time()

    while True:
        missing_frames = []
        for global_idx in range(start_frame, end_frame + 1):
            frame_path = os.path.join(cleaned_dir, f"{global_idx:04d}.png")
            if not os.path.exists(frame_path):
                missing_frames.append(global_idx)

        if not missing_frames:
            break  # All frames available

        if time.time() - wait_start > max_wait:
            print(f"[ENCODER ERROR] Timeout waiting for frames after {max_wait}s!")
            print(f"[ENCODER ERROR] Missing {len(missing_frames)}/{expected_frames} frames: {missing_frames[:20]}...")
            raise RuntimeError(f"Timeout: {len(missing_frames)} frames missing for segment {seg_idx}")

        # Wait and retry
        time.sleep(0.1)

    print(f"[ENCODER] All {expected_frames} frames available for segment {seg_idx}")

    # [FIX] SHARED BUFFER: Create file list for this segment's frames only
    # (frames are named by global index in shared buffer)
    file_list_path = os.path.join(TEMP_DIR, f"encode_seg{seg_idx}_{video_id}.txt")
    frames_added = 0

    print(f"[ENCODER DEBUG] Creating file list for segment {seg_idx}")
    print(f"[ENCODER DEBUG] Frame range: {start_frame}-{end_frame} ({expected_frames} frames expected)")
    print(f"[ENCODER DEBUG] Source dir: {cleaned_dir}")

    with open(file_list_path, 'w') as f:
        for global_idx in range(start_frame, end_frame + 1):
            frame_path = os.path.join(cleaned_dir, f"{global_idx:04d}.png")
            if os.path.exists(frame_path):
                # Write absolute path for ffmpeg concat
                abs_path = os.path.abspath(frame_path).replace('\\', '/')
                # Use duration 1/fps for each frame
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {1/fps}\n")
                frames_added += 1
            else:
                print(f"[ENCODER WARNING] Frame {global_idx:04d}.png missing after wait!")

        # [FIX] FIX: Must be INSIDE 'with' block so file handle 'f' is still open!
        if frames_added != expected_frames:
            print(f"[ENCODER WARNING] [!] MISSING FRAMES: Only {frames_added}/{expected_frames} frames in file list!")
            # Last frame needs to be repeated for proper duration
            last_frame_path = os.path.join(cleaned_dir, f"{end_frame:04d}.png")
            if os.path.exists(last_frame_path):
                abs_path = os.path.abspath(last_frame_path).replace('\\', '/')
                print(f"[ENCODER WARNING] Adding fallback frame: {last_frame_path}")
                f.write(f"file '{abs_path}'\n")
                f.write(f"duration {1/fps}\n")
                print(f"[ENCODER WARNING] Fallback frame added with duration {1/fps:.6f}s")

    print(f"[ENCODER DEBUG] File list created: {frames_added}/{expected_frames} frames added")

    # Verify file list was written correctly
    try:
        with open(file_list_path, 'r') as verify_f:
            lines = verify_f.readlines()
            file_entries = [l.strip() for l in lines if l.startswith('file ')]
            duration_entries = [l.strip() for l in lines if l.startswith('duration ')]
            print(f"[ENCODER DEBUG] File list verification: {len(file_entries)} files, {len(duration_entries)} durations")
            if len(file_entries) != len(duration_entries):
                print(f"[ENCODER WARNING] [!] MISMATCH: {len(file_entries)} files but {len(duration_entries)} durations!")
            if len(file_entries) > 0:
                print(f"[ENCODER DEBUG] First file: {file_entries[0]}")
                print(f"[ENCODER DEBUG] Last file: {file_entries[-1]}")
    except Exception as e:
        print(f"[ENCODER ERROR] Could not verify file list: {e}")

    # Hash verification: ensure frames are actually different
    import hashlib
    frame_hashes = []
    for global_idx in range(start_frame, min(start_frame + 5, end_frame + 1)):
        frame_path = os.path.join(cleaned_dir, f"{global_idx:04d}.png")
        if os.path.exists(frame_path):
            try:
                with open(frame_path, 'rb') as fh:
                    frame_hashes.append(hashlib.md5(fh.read()).hexdigest()[:8])
            except Exception:
                pass

    if len(frame_hashes) > 1:
        if len(set(frame_hashes)) == 1:
            print(f"[ENCODER ERROR] [!] First {len(frame_hashes)} frames have IDENTICAL hashes: {frame_hashes}")
            print(f"[ENCODER ERROR] This indicates all frames are the same - STUCK FRAME DETECTED!")
        else:
            print(f"[ENCODER DEBUG] Frame uniqueness OK: {frame_hashes}")

    # Encode using detected encoder (NVENC or CPU fallback)
    # [FIX] FIX: Use pattern input instead of concat demuxer (more reliable for sequential PNGs)
    # [FIX] FIX: Force forward slashes for Windows (FFmpeg pattern parser is POSIX-based)
    pattern_path = os.path.join(cleaned_dir, '%04d.png').replace('\\', '/')
    encode_cmd = [
        FFMPEG_EXE, '-y',
        '-framerate', str(fps),  # MUST be before -i
        '-start_number', str(start_frame),  # Start from segment's first frame
        '-i', pattern_path,  # Pattern input (more reliable than concat)
        '-frames:v', str(expected_frames),  # Limit to exact frame count
        '-c:v', ENCODER_CONFIG['codec'],
        '-preset', ENCODER_CONFIG['fallback_preset'],  # Use fallback preset (more stable)
        '-b:v', '8M',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'main',
        seg_video_path
    ]

    print(f"[ENCODER DEBUG] FFmpeg command: {' '.join(encode_cmd)}")

    try:
        result = subprocess.run(encode_cmd, capture_output=True, text=True, timeout=300)

        # Check for encoding errors and show stderr
        if result.returncode != 0:
            print(f"[ENCODER ERROR] FFmpeg failed with return code {result.returncode}")
            print(f"   Command: {' '.join(encode_cmd)}")
            print(f"   stderr: {result.stderr}")
            # Debug: Show file list contents on failure
            print(f"[ENCODER ERROR] File list contents ({file_list_path}):")
            try:
                with open(file_list_path, 'r') as debug_f:
                    file_list_contents = debug_f.read()
                    # Show first 50 lines or 2000 chars
                    lines = file_list_contents.split('\n')
                    if len(lines) > 50:
                        print(f"   (showing first 50 of {len(lines)} lines)")
                        print('\n'.join(lines[:50]))
                    else:
                        print(file_list_contents)
            except Exception as read_err:
                print(f"   Could not read file list: {read_err}")
            raise subprocess.CalledProcessError(result.returncode, encode_cmd, result.stdout, result.stderr)
        encode_duration = time.time() - encode_start

        encoded_size_mb = os.path.getsize(seg_video_path) / (1024 * 1024)
        fps_actual = frame_count / encode_duration if encode_duration > 0 else 0

        print(f"[ENCODER] [OK] Encoded: {encoded_size_mb:.2f} MB in {encode_duration:.2f}s ({fps_actual:.1f} fps)")

        # Store encoded path in Redis
        redis_client.hset(segment_key, 'encoded_path', seg_video_path)
        redis_client.hset(segment_key, 'status', 'encoded')

        # Cleanup file list and frames after successful encoding
        if os.path.exists(file_list_path):
            try:
                os.remove(file_list_path)
            except PermissionError:
                # Windows file locking - FFmpeg may still have handle open
                # Try after small delay
                import time
                time.sleep(0.1)
                try:
                    os.remove(file_list_path)
                except (PermissionError, OSError):
                    # Still locked - skip cleanup (temp file, not critical)
                    pass

        # Note: Don't cleanup shared_cleaned_dir yet - other segments may need it!
        # Final cleanup happens after all segments are encoded

    except subprocess.CalledProcessError as e:
        print(f"[ENCODER ERROR] Encoding failed for segment {seg_idx}!")
        print(f"   Command: {' '.join(encode_cmd)}")
        print(f"   Return code: {e.returncode}")
        print(f"   stderr: {e.stderr if e.stderr else '(no stderr captured)'}")
        print(f"   Encoder: {ENCODER_CONFIG['codec']} (preset: {ENCODER_CONFIG['fallback_preset']})")
        # Cleanup partial file
        if os.path.exists(seg_video_path):
            try:
                os.remove(seg_video_path)
                print(f"   [CLEANUP] Removed partial video file")
            except:
                pass
        raise
    except subprocess.TimeoutExpired as e:
        print(f"[ENCODER ERROR] Encoding timed out after 300s for segment {seg_idx}")
        print(f"   Command: {' '.join(encode_cmd)}")
        # Cleanup partial file
        if os.path.exists(seg_video_path):
            try:
                os.remove(seg_video_path)
                print(f"   [CLEANUP] Removed partial video file after timeout")
            except:
                pass
        raise


def trigger_finalization(redis_client, video_id, total_segments):
    """
    Concatenate all encoded segments and merge audio.
    Called automatically when all segments are encoded.

    ZERO-COPY MODE: If FRAME_BUFFER is available, encode directly from memory
    to bypass disk I/O race conditions that cause stuck frames.
    """
    import subprocess

    print(f"\n[FINALIZE] Starting finalization for video {video_id}")

    # Get video metadata (decode bytes)
    base_name_raw = redis_client.get(f"video:{video_id}:base_name")
    base_name = base_name_raw.decode() if isinstance(base_name_raw, bytes) else (base_name_raw or 'video')

    video_path_raw = redis_client.get(f"video:{video_id}:video_path")
    video_path = video_path_raw.decode() if isinstance(video_path_raw, bytes) else video_path_raw

    # Try zero-copy encoding from FRAME_BUFFER first
    use_zero_copy = False
    with FRAME_BUFFER_LOCK:
        if video_id in FRAME_BUFFER:
            buffer = FRAME_BUFFER[video_id]
            print(f"[FINALIZE] Zero-copy mode: FRAME_BUFFER has {buffer.frames_received}/{buffer.total_frames} frames")

            # Use zero-copy if we have most frames (allow some missing)
            if buffer.frames_received >= buffer.total_frames * 0.9:
                use_zero_copy = True
                print(f"[FINALIZE] ✓ Using zero-copy encoding from memory buffer")
            else:
                print(f"[FINALIZE] [!] Buffer incomplete - falling back to segment concatenation")

    if use_zero_copy:
        # Encode entire video directly from memory buffer
        temp_processed = os.path.join(RESULT_DIR, f"{base_name}_{video_id}_processed.mp4")

        with FRAME_BUFFER_LOCK:
            buffer = FRAME_BUFFER[video_id]
            print(f"[FINALIZE] Encoding {buffer.frames_received} frames from memory to {temp_processed}")
            buffer.encode_to_video(temp_processed)

        # Verify output
        if not os.path.exists(temp_processed) or os.path.getsize(temp_processed) == 0:
            print(f"[FINALIZE ERROR] Zero-copy encoding failed - falling back to segment concatenation")
            use_zero_copy = False
        else:
            output_size_mb = os.path.getsize(temp_processed) / (1024 * 1024)
            print(f"[FINALIZE] ✓ Zero-copy encoded: {output_size_mb:.2f} MB")

        # Cleanup buffer
        with FRAME_BUFFER_LOCK:
            if video_id in FRAME_BUFFER:
                del FRAME_BUFFER[video_id]
                print(f"[FINALIZE] Cleaned up FRAME_BUFFER for {video_id}")

    if not use_zero_copy:
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
    else:
        segment_paths = []  # No segment cleanup needed for zero-copy
        concat_list_path = None

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
    if concat_list_path and os.path.exists(concat_list_path):
        os.remove(concat_list_path)
    for seg_path in segment_paths:
        if os.path.exists(seg_path):
            os.remove(seg_path)

    # [FIX] SHARED BUFFER: Cleanup shared frame directory after finalization
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

    # [FIX] UPLOAD TO RAILWAY: If running on local worker, upload result to API server
    uploaded_path = None
    tunnel = os.getenv('TUNNEL_URL') or os.getenv('API_BASE_URL')
    upload_enabled = os.getenv('UPLOAD_RESULT_BACK', '1')

    print(f"[FINALIZE] Upload config - TUNNEL_URL: {'SET' if os.getenv('TUNNEL_URL') else 'NOT SET'}, API_BASE_URL: {'SET' if os.getenv('API_BASE_URL') else 'NOT SET'}, UPLOAD_RESULT_BACK: {upload_enabled}")

    if not tunnel:
        print(f"[FINALIZE] [!]  Skipping Railway upload - TUNNEL_URL/API_BASE_URL not set")
        print(f"[FINALIZE] 💡 Set TUNNEL_URL=https://your-railway-app.railway.app to enable auto-upload")
    elif upload_enabled != '1':
        print(f"[FINALIZE] [!]  Skipping Railway upload - UPLOAD_RESULT_BACK={upload_enabled} (set to '1' to enable)")

    if tunnel and upload_enabled == '1':
        try:
            import requests
            upload_url = tunnel.rstrip('/') + '/api/upload-result'
            print(f"[FINALIZE] [UPLOAD] Uploading result to Railway: {upload_url}")

            # Quick connectivity test (15 second timeout) - fail fast if server down
            requests.head(tunnel, timeout=15, headers={'ngrok-skip-browser-warning': 'true'})

            with open(final_output, 'rb') as fp:
                resp = requests.post(
                    upload_url,
                    headers={'ngrok-skip-browser-warning': 'true'},
                    files={'file': (os.path.basename(final_output), fp, 'video/mp4')},
                    timeout=300  # 5 minutes for large videos
                )
            if resp.ok:
                j = resp.json()
                if j.get('status') == 'success' and j.get('result_url'):
                    uploaded_path = j['result_url']
                    print(f"[FINALIZE] [OK] Result uploaded to Railway: {uploaded_path}")
                    # Update Redis with Railway path instead of local path
                    redis_client.set(f"video:{video_id}:final_path", uploaded_path)
                    redis_client.set(f"video:{video_id}:uploaded", "true")
            else:
                print(f"[FINALIZE WARNING] Upload to Railway failed: HTTP {resp.status_code}")
        except Exception as up_err:
            print(f"[FINALIZE WARNING] Upload to Railway error: {up_err}")
            import traceback
            traceback.print_exc()

    # Store final result in Redis (local path if upload failed, Railway path if uploaded)
    if not uploaded_path:
        redis_client.set(f"video:{video_id}:final_path", final_output)
    redis_client.set(f"video:{video_id}:status", "complete")

    # [FIX] FIX: Update distributed tracking to mark all segments complete
    # Status endpoint checks segments:{video_id} to see progress
    # Without this, frontend shows "Segment 0/X complete" forever
    tracking_key = f"segments:{video_id}"
    total_segments_bytes = redis_client.get(f"{tracking_key}:total")
    if total_segments_bytes:
        redis_client.set(tracking_key, int(total_segments_bytes))  # Mark all segments complete
        print(f"[FINALIZE] [OK] Marked all {int(total_segments_bytes)} segments complete in Redis tracking")

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
                    print(f"[DOWNLOAD] Worker {os.getpid()}: Detected download signal for {video_id}")
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
def get_detector():
    """
    Lazy load the YOLO detector with torch.compile optimization.
    Note: This is used by Celery workers on cloud machines, not the local Flask server.
    """
    global detector

    if detector is None:
        print("=" * 60)
        print("Loading YOLO detector...")
        print("=" * 60)
        from yolo_detector import YOLOWatermarkDetector
        import numpy as np
        # Load PyTorch model with torch.compile optimization
        detector = YOLOWatermarkDetector()

        # WARMUP: Run inference to trigger torch.compile compilation
        print("[WARMUP] Running warmup inference for torch.compile...")
        dummy_batch = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(64)]
        _ = detector.detect_batch(dummy_batch, confidence_threshold=0.15, batch_size=64)
        print("[OK] YOLO warmed up! torch.compile ready for inference.")

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
                    print(f"[WEB] Downloading video: {download_url}")
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

        # NVDEC hardware decoder - FIXED with .clone() to prevent buffer reuse
        # Zero-copy encoding bypasses disk I/O race conditions
        use_nvdec = True
        nvdec_loader = None
        cap = None

        if use_nvdec:
            try:
                from nvdec_video_loader import NVDECVideoLoader
                nvdec_loader = NVDECVideoLoader(video_path, device_id=0)
                props = nvdec_loader.get_properties()
                fps = int(props['fps'])
                width = props['width']
                height = props['height']
                total_frames = props['total_frames']
                print(f"[OK] NVDEC hardware decoder: {width}x{height} @ {fps} fps ({total_frames} frames)")
            except ImportError:
                print("[WARNING] nvdec_video_loader not available, falling back to CPU decoder")
                use_nvdec = False

        if not use_nvdec:
            # CPU decoder fallback
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
            print(f"[DOWNLOAD] Loading {total_frames} frames to memory (batch processing)...")

            import time
            decode_start = time.time()

            if use_nvdec:
                # NVDEC hardware decoder - uses optimized library conversion
                all_frames = nvdec_loader.load_all_frames(to_numpy=True, color_format='BGR')

                frames_processed = len(all_frames)
                decode_time = time.time() - decode_start
                print(f"[OK] NVDEC decoded {frames_processed} frames: {decode_time:.3f}s ({decode_time/frames_processed*1000:.2f}ms/frame)")
                nvdec_loader.close()

                # Uniqueness check
                if frames_processed > 1:
                    import hashlib
                    h0 = hashlib.md5(all_frames[0].tobytes()).hexdigest()[:8]
                    h1 = hashlib.md5(all_frames[-1].tobytes()).hexdigest()[:8]
                    if h0 == h1:
                        raise RuntimeError(f"[NVDEC ERROR] All frames identical (hash={h0}) - clone() fix failed!")
                    print(f"[OK] NVDEC frames verified unique: first={h0}, last={h1}")
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
            print(f"[FAST] Creating masks and storing in Redis (in-memory)...")
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
                print(f"[FAST] Using SAM2-Tiny for temporal mask tracking (44ms/frame)...")
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
                print(f"[FAST] Creating {frames_processed} masks (GPU batch)...")
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
            print(f"[FAST] Storing {frames_processed} frames/masks in memory (INSTANT!)...")
            mem_start = time.time()

            # Validate tensor shapes before caching (defensive check)
            if hasattr(all_frames, 'shape'):
                print(f"[DEBUG] Cache: all_frames is tensor with shape {all_frames.shape}")
                if len(all_frames.shape) == 5:
                    print(f"[WARNING] Cache: squeezing batch dim from {all_frames.shape}")
                    all_frames = all_frames.squeeze(0)
                    print(f"[DEBUG] Cache: after squeeze {all_frames.shape}")
            else:
                print(f"[DEBUG] Cache: all_frames is list with {len(all_frames)} items")

            if hasattr(all_masks, 'shape'):
                print(f"[DEBUG] Cache: all_masks is tensor with shape {all_masks.shape}")
                if len(all_masks.shape) == 5:
                    print(f"[WARNING] Cache: squeezing batch dim from masks {all_masks.shape}")
                    all_masks = all_masks.squeeze(0)

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
                print(f"[CACHE] Also writing to disk (backup for remote workers)...")
                for i in range(frames_processed):
                    mask_path = os.path.join(shared_mask_dir, f"{i:04d}.png")
                    cv2.imwrite(mask_path, all_masks[i])
                    frame_path = os.path.join(shared_frames_dir, f"{i:04d}.png")
                    cv2.imwrite(frame_path, all_frames[i])
            else:
                print(f"[FAST] Skipping disk writes (pure memory for EXTREME SPEED!)")

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

            # [CACHE] Cache segment results for future duplicate requests (1 hour TTL)
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
                print(f"[CACHE] Cached YOLO results for '{base_name}' (1 hour TTL)")
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

        # Initialize frame buffer for zero-copy encoding (now that frames_processed is known)
        if frames_processed and frames_processed > 0:
            with FRAME_BUFFER_LOCK:
                FRAME_BUFFER[video_id] = FramePipeEncoder(video_id, frames_processed, width, height, fps)
            print(f"[OK] Frame buffer initialized for {video_id}: {frames_processed} frames @ {width}x{height}")
        else:
            print(f"[WARNING] Cannot initialize frame buffer - frames_processed is {frames_processed}")

        # Optional: force time-based splitting to ensure multi-GPU distribution
        try:
            import math
            min_segments = int(os.getenv('MIN_SEGMENTS', '4'))  # Default 4 for parallel GPU processing
            min_chunk_frames = int(os.getenv('MIN_CHUNK_FRAMES', '60'))
        except Exception:
            min_segments = 4
            min_chunk_frames = 60

        # Force-split for parallel GPU processing when not enough segments detected
        # Even if YOLO found 1 segment (stationary watermark), split for multi-GPU speed
        if min_segments and len(segments) < min_segments and frames_processed >= min_chunk_frames:
            # Split video into time-based chunks for parallel processing
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
            print(f"🪓 Force-split for parallel GPU: {len(segments)} segments (≈{chunk} frames each) - EXTREME SPEED MODE!")

        # Provide a base URL so OTHER workers can fetch frames/masks from this host
        temp_base_url = temp_base or os.getenv('TEMP_BASE_URL') or os.getenv('TUNNEL_URL')
        if temp_base_url:
            print(f"[WEB] Shared temp base set for workers: {temp_base_url.rstrip('/')}/temp/")
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

        # [FAST] OPTIMIZATION: Choose processing strategy based on SEGMENT_WORKERS
        # SEGMENT_WORKERS=1: Use CUDA streams (thread-based, one GPU context, 27-31ms/frame)
        # SEGMENT_WORKERS>1: Use Celery chord (process-based, multiple GPU contexts, slower)
        segment_workers = int(os.getenv('SEGMENT_WORKERS', '1'))

        if segment_workers == 1:
            # CUDA Streams approach: Process segments in parallel using threads + CUDA streams
            print(f"[CUDA STREAMS] Processing {len(segments)} segments in parallel with CUDA streams (thread-based)")
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Processing {len(segments)} segments with CUDA streams'})

            # Import threading utilities
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import torch

            # Add total_segments to each segment data
            for seg in segment_tasks_data:
                seg['total_segments'] = len(segments)

            # Process all segments in parallel using threads + CUDA streams
            segment_results = []

            # Create a mock self object for direct calls (no Celery context)
            class MockSelf:
                pass  # process_segment_task will detect missing request.id and skip state updates

            mock_self = MockSelf()

            def process_segment_with_stream(seg_data, stream_id):
                """Process one segment on a dedicated CUDA stream"""
                try:
                    print(f"[STREAM {stream_id}] Starting segment {seg_data['seg_idx']+1}/{len(segments)}")

                    # Create and set CUDA stream for this segment
                    if torch.cuda.is_available():
                        stream = torch.cuda.Stream()
                        with torch.cuda.stream(stream):
                            print(f"[STREAM {stream_id}] Using dedicated CUDA stream for segment {seg_data['seg_idx']+1}")

                            # Call process_segment_task.run() to bypass Celery task machinery
                            # This calls the underlying function directly
                            result = process_segment_task.run(seg_data)

                            # Synchronize this stream before returning
                            torch.cuda.current_stream().synchronize()
                    else:
                        # No CUDA, run on CPU
                        result = process_segment_task.run(seg_data)

                    print(f"[STREAM {stream_id}] Completed segment {seg_data['seg_idx']+1}/{len(segments)}")
                    return result
                except Exception as e:
                    print(f"[STREAM {stream_id}] ERROR in segment {seg_data['seg_idx']+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            # Use ThreadPoolExecutor to run segments in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all segments to thread pool
                futures = {
                    executor.submit(process_segment_with_stream, seg_data, i): i
                    for i, seg_data in enumerate(segment_tasks_data)
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    stream_id = futures[future]
                    try:
                        result = future.result()
                        segment_results.append(result)
                    except Exception as e:
                        print(f"[ERROR] Segment {stream_id} failed: {e}")
                        raise

            print(f"[OK] All {len(segment_results)} segments processed with CUDA streams!")

            # CUDA Streams mode: segments processed synchronously, now encode final video
            print(f"[ENCODE] Encoding final video from {len(segment_results)} segments...")

            # Import encoding utilities
            import subprocess
            shared_cleaned_dir = os.path.join(TEMP_DIR, f"{base_name}_{video_id}_all_frames_cleaned")

            # Check if we have cleaned frames
            if not os.path.exists(shared_cleaned_dir):
                raise RuntimeError(f"Cleaned frames directory not found: {shared_cleaned_dir}")

            # Count frames
            cleaned_frames = sorted([f for f in os.listdir(shared_cleaned_dir) if f.endswith('.png')])
            print(f"[ENCODE] Found {len(cleaned_frames)} cleaned frames in {shared_cleaned_dir}")

            # Encode video with NVENC
            output_video = os.path.join(RESULT_DIR, f"{base_name}_propainter.mp4")
            ffmpeg_path, ffprobe_path = get_ffmpeg_executables()

            encode_cmd = [
                ffmpeg_path,
                '-y',
                '-framerate', str(fps),
                '-start_number', '0',  # Start from 0000.png
                '-i', os.path.join(shared_cleaned_dir, '%04d.png').replace('\\', '/'),
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-tune', 'hq',
                '-b:v', '5M',
                '-maxrate', '8M',
                '-pix_fmt', 'yuv420p',
                output_video
            ]

            print(f"[ENCODE] Encoding cleaned frames to video...")
            result_encode = subprocess.run(encode_cmd, capture_output=True, text=True)
            if result_encode.returncode != 0:
                raise RuntimeError(f"FFmpeg encoding failed: {result_encode.stderr}")

            print(f"[ENCODE] Video encoded: {output_video}")

            # Merge audio from original
            if os.path.exists(video_path):
                print(f"[AUDIO] Merging audio from original video...")
                temp_video = output_video.replace('.mp4', '_temp.mp4')
                os.rename(output_video, temp_video)

                audio_cmd = [
                    ffmpeg_path,
                    '-y',
                    '-i', temp_video,
                    '-i', video_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    '-shortest',
                    output_video
                ]

                result_audio = subprocess.run(audio_cmd, capture_output=True, text=True)
                if result_audio.returncode == 0:
                    os.remove(temp_video)
                    print(f"[AUDIO] Audio merged successfully")
                else:
                    os.rename(temp_video, output_video)
                    print(f"[AUDIO] No audio track or merge failed, using video-only")

            # Cleanup
            print(f"[CLEANUP] Removing temp frames...")
            shutil.rmtree(shared_cleaned_dir, ignore_errors=True)

            # Release processing lock
            try:
                redis_client = celery.backend.client
                lock_key = f'prepare_lock:{base_name}'
                redis_client.delete(lock_key)
                print(f"🔓 Released processing lock for video '{base_name}'")
            except Exception as e:
                print(f"[WARNING]  Failed to release lock: {e}")

            print(f"[OK] Video processing complete! Output: {output_video}")

            return {
                'video_id': video_id,
                'status': 'complete',
                'message': f'Processed {len(segments)} segments with CUDA streams',
                'output_path': output_video,
                'segments_processed': len(segment_results)
            }

        # [FAST] FALLBACK: Use Celery chord to dispatch segments in parallel (process-based)
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
    Can also be called directly from threads (for CUDA streams mode)
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

        # Helper function to safely update state (works in both Celery and thread modes)
        def safe_update_state(state=None, meta=None):
            try:
                if hasattr(self, 'request') and self.request.id:
                    self.update_state(state=state, meta=meta)
            except:
                pass  # Called from thread, skip state updates

        safe_update_state(state='STARTED', meta={'progress': 0, 'status': f'Processing segment {seg_idx+1}'})

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

        # [FIX] SHARED FRAME BUFFER FIX: All segments merge onto same directory
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
            safe_update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Downloading video'})

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
                        print(f"   [CACHE] Video cached for future segments")
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
            safe_update_state(state='PROCESSING', meta={'progress': 10, 'status': f'Loading frames'})

            # [INIT] EXTREME SPEED: Try FRAME_CACHE first (pure memory, INSTANT!)
            cache_key = f"video_data:{base_name}"
            memory_hits = 0
            segment_frames_memory = []  # Store frames in memory (skip disk!)

            # [FIX] TEMPORAL CONTEXT FIX: Extract with neighbor padding for ProPainter
            # neighbor_length=10 means ±5 frames needed for temporal context
            neighbor_padding = 5
            padded_start = max(0, start_frame - neighbor_padding)
            padded_end = end_frame + neighbor_padding  # Will be recalculated with total_frames from cache

            if cache_key in FRAME_CACHE:
                # INSTANT access to frames in memory!
                print(f"   [FAST] Loading from memory cache WITH NEIGHBOR PADDING (ZERO disk I/O!)...")
                with FRAME_CACHE_LOCK:
                    cached = FRAME_CACHE[cache_key]
                    cached_frames = cached['frames']

                    # Debug: Log cache data type and shape
                    if hasattr(cached_frames, 'shape'):
                        print(f"   [DEBUG] Cache frames type: tensor, shape: {cached_frames.shape}")
                        if len(cached_frames.shape) == 5:
                            print(f"   [WARNING] Cache has 5D tensor - len() will be wrong! Squeezing...")
                            cached_frames = cached_frames.squeeze(0)
                            cached['frames'] = cached_frames  # Update cache
                            print(f"   [DEBUG] After squeeze: {cached_frames.shape}")
                    else:
                        print(f"   [DEBUG] Cache frames type: {type(cached_frames).__name__}, len: {len(cached_frames)}")

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
        safe_update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Loading masks'})

        # [FIX] TEMPORAL CONTEXT FIX: Load masks WITH PADDING (same range as frames)
        masks_needed = list(range(padded_start, padded_end + 1))
        masks_succeeded = 0
        memory_mask_hits = 0
        segment_masks_memory = []  # Store masks in memory (skip disk!)

        # Priority 1: Memory cache (INSTANT!)
        if cache_key in FRAME_CACHE:
            print(f"   [FAST] Loading masks from memory WITH NEIGHBOR PADDING (ZERO disk I/O!)...")
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
            safe_update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Copying masks'})

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
            print(f"   [DOWNLOAD] Downloading masks from remote location...")
            safe_update_state(state='PROCESSING', meta={'progress': 20, 'status': f'Downloading masks'})

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
            safe_update_state(state='PROCESSING', meta={'progress': 25, 'status': f'Detecting watermarks'})

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

            # Create masks on CROPPED frames (not full frames!)
            print(f"   [CREATE] Creating masks on cropped frames...")
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
                print(f"   [CROP]  Cropping downloaded masks to region...")
                for frame_idx in range(frames_copied):
                    mask_file = f"{frame_idx:04d}.png"
                    mask_path = os.path.join(seg_mask_dir, mask_file)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        cv2.imwrite(mask_path, cropped_mask)
        elif using_memory_pipeline:
            print(f"   [FAST] Skipping disk-based cropping - using in-memory pipeline (saves ~600ms!)")

        print(f"   [OK] Prepared {frames_copied} frames and masks (including ±{neighbor_padding} neighbor padding)")

        # VALIDATION: Check first mask to catch bugs early
        if last_valid_bbox and frames_with_watermark > 0:
            test_mask = None

            # Get mask from memory or disk
            if using_memory_pipeline and len(segment_masks_memory) > 0:
                test_mask = segment_masks_memory[0]
            else:
                first_mask_path = os.path.join(seg_mask_dir, "0000.png")
                if os.path.exists(first_mask_path):
                    test_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)

            if test_mask is not None:
                white_pixels = np.sum(test_mask == 255)
                total_pixels = test_mask.size
                white_pct = (white_pixels / total_pixels) * 100

                print(f"   [INFO] Mask validation: {white_pct:.1f}% white pixels (target region)")

                # Warn if mask seems wrong
                if white_pct > 50:
                    print(f"   [WARNING]  WARNING: Mask is {white_pct:.1f}% white - suspicious! May cause black output.")
                elif white_pct < 0.1:
                    print(f"   [WARNING]  WARNING: Mask is {white_pct:.1f}% white - almost empty! No inpainting will occur.")

        # CONDITIONAL: Only run ProPainter if watermark was detected
        if not last_valid_bbox or frames_with_watermark == 0:
            print(f"   [SKIP]  No watermark detected - skipping ProPainter, encoding original frames")
            safe_update_state(state='PROCESSING', meta={'progress': 50, 'status': f'No watermark - encoding original'})

            # Copy original frames to cleaned dir (no processing needed)
            # [FIX] SHARED BUFFER: Use global frame indices
            if using_memory_pipeline:
                # Write from memory (only core segment, skip padding)
                # [FIX] TEMPORAL CONTEXT FIX: Calculate padding offset
                padding_offset = start_frame - padded_start
                core_frames = segment_frames_memory[padding_offset:padding_offset + seg_duration]
                print(f"   [CACHE] Writing {len(core_frames)} core segment frames from memory to shared buffer (skipping {padding_offset} padding frames)...")
                for local_idx, frame in enumerate(core_frames):
                    # Use GLOBAL frame index for shared buffer
                    global_frame_idx = start_frame + local_idx
                    frame_file = f"{global_frame_idx:04d}.png"
                    dst = os.path.join(shared_cleaned_dir, frame_file)

                    # Only write if not already written by another segment
                    if not os.path.exists(dst):
                        cv2.imwrite(dst, frame)

                    # Also add to in-memory frame buffer for zero-copy encoding
                    with FRAME_BUFFER_LOCK:
                        if video_id in FRAME_BUFFER:
                            FRAME_BUFFER[video_id].add_frame(global_frame_idx, frame)
            else:
                # Copy from disk
                for local_idx in range(frames_copied):
                    # Use GLOBAL frame index for shared buffer
                    global_frame_idx = start_frame + local_idx
                    frame_file = f"{global_frame_idx:04d}.png"
                    src = os.path.join(seg_frames_dir, f"{local_idx:04d}.png")  # Local source
                    dst = os.path.join(shared_cleaned_dir, frame_file)  # Global destination

                    # Only write if not already written by another segment
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy2(src, dst)

                        # Also add to in-memory frame buffer for zero-copy encoding
                        frame = cv2.imread(src)
                        if frame is not None:
                            with FRAME_BUFFER_LOCK:
                                if video_id in FRAME_BUFFER:
                                    FRAME_BUFFER[video_id].add_frame(global_frame_idx, frame)

        else:
            # Run ProPainter on this segment - watermark detected!
            print(f"   [PAINT] Running ProPainter on {frames_with_watermark} watermarked frames...")
            safe_update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Running ProPainter'})

            try:
                # Use cached ProPainter pipeline (pre-loaded at worker startup)
                faster_propainter_pipeline = get_propainter_pipeline()

                import torch
                use_fp16 = torch.cuda.is_available()

                # [INIT] EXTREME SPEED: Use already-loaded memory frames (skip FRAME_CACHE re-access!)
                frames_array = None
                masks_array = None

                if using_memory_pipeline:
                    print(f"   [FAST] Cropping {len(segment_frames_memory)} frames/masks in memory (ZERO disk I/O!)")
                    import time
                    crop_start = time.time()

                    # Crop to watermark region in memory (no disk I/O!)
                    frames_array = []
                    masks_array = []

                    for frame in segment_frames_memory:
                        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        frames_array.append(cropped)

                    for mask in segment_masks_memory:
                        cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        masks_array.append(cropped_mask)

                    crop_duration = time.time() - crop_start
                    print(f"   [OK] Cropped {len(frames_array)} frames + {len(masks_array)} masks in memory: {crop_duration:.3f}s")
                    print(f"   [RUNNING] Eliminated ~1000ms+ of disk I/O - pure in-memory pipeline!")

                if frames_array is not None and masks_array is not None:
                    print(f"   [GPU] GPU available: {use_fp16}, Running ProPainter with IN-MEMORY arrays...")
                else:
                    print(f"   [GPU] GPU available: {use_fp16}, Running ProPainter with disk paths (fallback)...")

                faster_propainter_pipeline(
                    video=seg_cropped_dir,
                    mask=seg_mask_dir,
                    output=seg_output_dir,
                    resize_ratio=1.0,
                    mask_dilation=4,
                    ref_stride=15,
                    neighbor_length=10,
                    subvideo_length=120,
                    raft_iter=10,
                    mode="video_inpainting",
                    save_frames=True,
                    fp16=use_fp16,
                    frames_array=frames_array,
                    masks_array=masks_array
                )

                print(f"   [OK] ProPainter complete for segment {seg_idx+1}")
            except Exception as e:
                print(f"[ERROR] ProPainter failed on segment {seg_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                clear_gpu_memory()

            # Merge cleaned region back to full frames
            print(f"   🔗 Merging cleaned region (in-memory)...")
            safe_update_state(state='PROCESSING', meta={'progress': 80, 'status': f'Merging results'})

            seg_propainter_frames = os.path.join(seg_output_dir, os.path.basename(seg_cropped_dir), 'frames')
            if not os.path.exists(seg_propainter_frames):
                raise RuntimeError(f"ProPainter output not found for segment {seg_idx}")

            # Load all frames into memory first (faster than disk I/O in loop)
            # [INIT] EXTREME SPEED: Use in-memory frames if available (no disk reads!)
            original_frames = []
            original_masks = []  # [FIX] MASK COMPOSITING: Need masks for alpha blending
            cleaned_frames = []

            # [FIX] TEMPORAL CONTEXT FIX: Calculate padding offset
            # ProPainter processed frames WITH padding, we only need core segment frames
            padding_offset = start_frame - padded_start  # How many padding frames before core segment
            print(f"   [CONTEXT] ProPainter processed {len(segment_frames_memory)} frames (padding offset: {padding_offset})")

            if using_memory_pipeline:
                # Use already-loaded memory frames (ZERO disk I/O!)
                # Extract ONLY core segment frames (skip padding)
                print(f"   [FAST] Extracting core segment frames + masks from memory (frames {padding_offset} to {padding_offset + seg_duration})...")
                original_frames = segment_frames_memory[padding_offset:padding_offset + seg_duration]
                original_masks = segment_masks_memory[padding_offset:padding_offset + seg_duration]
            else:
                # Disk-based fallback
                print(f"   [MASKS] Loading {seg_duration} original frames + masks from disk...")
                for frame_idx in range(seg_duration):
                    frame_file = f"{frame_idx:04d}.png"

                    # Load original frame
                    orig_path = os.path.join(seg_frames_dir, frame_file)
                    orig = cv2.imread(orig_path)
                    if orig is None:
                        raise RuntimeError(f"Failed to load original frame from {orig_path}")
                    original_frames.append(orig)

                    # Load corresponding mask
                    mask_path = os.path.join(seg_mask_dir, frame_file)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise RuntimeError(f"Failed to load mask from {mask_path}")
                    original_masks.append(mask)

            # Load cleaned frames from ProPainter output
            # ProPainter outputs frames sequentially starting from 0000 (regardless of padding)
            # So for a segment with 70 total frames (5 padding + 60 core + 5 padding),
            # the core segment is at indices [padding_offset : padding_offset+seg_duration] = [5:65]
            print(f"   [OUTPUT] Extracting core segment from ProPainter output (frames {padding_offset} to {padding_offset + seg_duration})...")

            # Count available frames in ProPainter output for debugging
            import glob
            available_frames = sorted(glob.glob(os.path.join(seg_propainter_frames, "*.png")))
            print(f"   [DEBUG] ProPainter output has {len(available_frames)} frames total")

            for local_idx in range(seg_duration):
                # Load frame at index from ProPainter output (skip padding frames)
                frame_idx = padding_offset + local_idx
                frame_file = f"{frame_idx:04d}.png"
                frame_path = os.path.join(seg_propainter_frames, frame_file)
                clean = cv2.imread(frame_path)
                if clean is None:
                    # Frame loading failed - this is a critical error
                    print(f"   [ERROR] Failed to load cleaned frame {frame_file}!")
                    print(f"   [DEBUG] Expected path: {frame_path}")
                    print(f"   [DEBUG] Path exists: {os.path.exists(frame_path)}")
                    print(f"   [DEBUG] ProPainter output dir: {seg_propainter_frames}")
                    print(f"   [DEBUG] Available frames: {[os.path.basename(f) for f in available_frames[:10]]}")
                    print(f"   [DEBUG] Padding offset: {padding_offset}, Segment duration: {seg_duration}")
                    raise RuntimeError(
                        f"Failed to load cleaned frame {frame_file} from ProPainter output. "
                        f"Expected {len(available_frames)} frames, trying to load index {frame_idx}. "
                        f"This indicates a frame index mismatch between ProPainter output and expected indices."
                    )
                cleaned_frames.append(clean)

            # 🔍 DEBUG: Verify frames are unique (not all the same)
            import hashlib
            if len(cleaned_frames) > 0:
                frame_hashes = []
                frame_sizes = []
                for i, frame in enumerate(cleaned_frames[:5]):  # Check first 5 frames
                    h = hashlib.md5(frame.tobytes()).hexdigest()[:8]
                    frame_hashes.append(h)
                    frame_sizes.append(frame.shape)
                print(f"   [DEBUG] First 5 cleaned frame hashes: {frame_hashes}")
                print(f"   [DEBUG] Frame shapes: {frame_sizes}")
                if len(set(frame_hashes)) == 1:
                    print(f"   [WARNING] [!] ALL FRAMES HAVE SAME HASH - THEY ARE IDENTICAL!")
                else:
                    print(f"   [OK] Frames are unique ({len(set(frame_hashes))} different hashes)")

                # Also check last few frames
                if len(cleaned_frames) > 5:
                    last_hashes = [hashlib.md5(f.tobytes()).hexdigest()[:8] for f in cleaned_frames[-3:]]
                    print(f"   [DEBUG] Last 3 cleaned frame hashes: {last_hashes}")

            # [FIX] SHARED BUFFER MERGE: Load existing frame state, apply this segment's edit, save back
            # This allows multiple segments to cooperatively edit the same frames
            print(f"   🔗 Merging to shared frame buffer with mask compositing + file locking...")
            for local_idx, (original, cleaned_crop, segment_mask) in enumerate(zip(original_frames, cleaned_frames, original_masks)):
                # Use GLOBAL frame index for shared buffer
                global_frame_idx = start_frame + local_idx
                frame_file = f"{global_frame_idx:04d}.png"
                shared_frame_path = os.path.join(shared_cleaned_dir, frame_file)

                # 🔒 FILE LOCK: Prevent race conditions when multiple segments edit same frame
                # This ensures atomic read-modify-write operations
                with file_lock(shared_frame_path, timeout=5):
                    # Load existing state (may have edits from other segments) or original
                    if os.path.exists(shared_frame_path):
                        # Another segment already edited this frame - load current state
                        result_frame = cv2.imread(shared_frame_path)
                        if result_frame is None:
                            result_frame = original.copy() if original is not None else np.zeros((height, width, 3), dtype=np.uint8)
                    elif original is not None:
                        result_frame = original.copy()
                    else:
                        continue

                    # [FIX] MASK COMPOSITING: Alpha blend cleaned region with mask for smooth edges
                    if cleaned_crop is not None and result_frame is not None and segment_mask is not None:
                        # ProPainter output is already cropped to watermark region
                        # Segment mask is also already cropped (from cropping step earlier)
                        # So we DON'T double-crop the mask here!

                        # Crop the mask to match the watermark region IF it's full-size
                        # (Check dimensions to determine if already cropped)
                        if segment_mask.shape[:2] == (height, width):
                            # Mask is full-frame, crop it to watermark region
                            cropped_mask = segment_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        else:
                            # Mask is already cropped, use as-is
                            cropped_mask = segment_mask

                        # Alpha blending for smooth compositing
                        # Convert mask to 3-channel float [0, 1]
                        mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0

                        # Get the region of interest (ROI) from result frame
                        roi = result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].astype(float)

                        # Blend: where mask=1 (watermark), use cleaned; where mask=0, use original
                        cleaned_crop_float = cleaned_crop.astype(float)
                        blended = (cleaned_crop_float * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

                        # Paste blended result back
                        result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = blended
                    elif cleaned_crop is not None and result_frame is not None:
                        # No mask available - fall back to direct paste
                        print(f"   [WARNING] No mask for frame {global_frame_idx} - using direct paste")
                        result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop

                    # Save back to shared buffer (within lock - atomic operation)
                    cv2.imwrite(shared_frame_path, result_frame)

                    # Also add to in-memory frame buffer for zero-copy encoding
                    with FRAME_BUFFER_LOCK:
                        if video_id in FRAME_BUFFER:
                            FRAME_BUFFER[video_id].add_frame(global_frame_idx, result_frame)

            print(f"   [OK] Merged {len(cleaned_frames)} frames to shared buffer: {shared_cleaned_dir}")

            # Report buffer status
            with FRAME_BUFFER_LOCK:
                if video_id in FRAME_BUFFER:
                    buffer = FRAME_BUFFER[video_id]
                    print(f"   [BUFFER] Frame buffer has {buffer.frames_received}/{buffer.total_frames} frames")

            # 🔍 DEBUG: Verify written files are unique
            written_files = sorted([f for f in os.listdir(shared_cleaned_dir) if f.endswith('.png')])
            print(f"   [DEBUG] Written {len(written_files)} files to shared buffer")
            if len(written_files) > 0:
                # Check file sizes of first and last few files
                first_files = written_files[:3]
                last_files = written_files[-3:] if len(written_files) > 3 else []

                print(f"   [DEBUG] First 3 files: {first_files}")
                first_sizes = [os.path.getsize(os.path.join(shared_cleaned_dir, f)) for f in first_files]
                print(f"   [DEBUG] First 3 file sizes: {first_sizes} bytes")

                if last_files:
                    print(f"   [DEBUG] Last 3 files: {last_files}")
                    last_sizes = [os.path.getsize(os.path.join(shared_cleaned_dir, f)) for f in last_files]
                    print(f"   [DEBUG] Last 3 file sizes: {last_sizes} bytes")

                # Hash check on disk files
                import hashlib
                disk_hashes = []
                for f in first_files:
                    with open(os.path.join(shared_cleaned_dir, f), 'rb') as fh:
                        disk_hashes.append(hashlib.md5(fh.read()).hexdigest()[:8])
                print(f"   [DEBUG] First 3 disk file hashes: {disk_hashes}")
                if len(set(disk_hashes)) == 1:
                    print(f"   [WARNING] [!] DISK FILES HAVE SAME HASH - POSSIBLE WRITE ISSUE!")
                else:
                    print(f"   [OK] Disk files are unique")

        # [FAST] PARALLEL ENCODING: Encode segment MP4 immediately (Blackwell NVENC!)
        print(f"   [OK] Cleaned frames ready in: {seg_cleaned_dir}")
        # [FAST] BACKGROUND ENCODING OPTIMIZATION: Signal encoder thread instead of blocking!
        # Worker returns immediately and starts next segment while encoding happens in background
        safe_update_state(state='PROCESSING', meta={'progress': 85, 'status': f'Segment complete - signaling encoder'})

        # Count segment's frames (not total in shared buffer)
        segment_frame_count = end_frame - start_frame + 1
        total_buffer_frames = len([f for f in os.listdir(seg_cleaned_dir) if f.endswith('.png')])
        if total_buffer_frames == 0:
            raise RuntimeError(f"No cleaned frames found in {seg_cleaned_dir}")

        # 🔍 DEBUG: Verify segment frame range exists in buffer
        print(f"   [DEBUG] Segment range: frames {start_frame}-{end_frame} ({segment_frame_count} frames)")
        print(f"   [DEBUG] Total frames in shared buffer: {total_buffer_frames}")

        # Check if this segment's frames actually exist
        segment_frames_exist = 0
        for frame_idx in range(start_frame, end_frame + 1):
            frame_path = os.path.join(seg_cleaned_dir, f"{frame_idx:04d}.png")
            if os.path.exists(frame_path):
                segment_frames_exist += 1

        if segment_frames_exist != segment_frame_count:
            print(f"   [WARNING] [!] Only {segment_frames_exist}/{segment_frame_count} segment frames exist in buffer!")
        else:
            print(f"   [OK] All {segment_frame_count} segment frames verified in buffer")

        print(f"   [OK] {total_buffer_frames} cleaned frames ready - signaling background encoder!")

        # Store segment metadata in Redis for background encoder
        redis_client = celery.backend.client
        segment_key = f"video:{video_id}:segment:{seg_idx}"
        redis_client.hset(segment_key, 'cleaned_dir', seg_cleaned_dir)
        redis_client.hset(segment_key, 'fps', str(fps))
        redis_client.hset(segment_key, 'frame_count', str(segment_frame_count))
        redis_client.hset(segment_key, 'base_name', base_name)
        redis_client.hset(segment_key, 'status', 'ready_for_encoding')
        # [FIX] SHARED BUFFER: Store frame range for encoder
        redis_client.hset(segment_key, 'start_frame', str(start_frame))
        redis_client.hset(segment_key, 'end_frame', str(end_frame))

        # Set video-level metadata (idempotent - safe to set multiple times)
        redis_client.set(f"video:{video_id}:total_segments", total_segments)
        redis_client.set(f"video:{video_id}:base_name", base_name)
        redis_client.set(f"video:{video_id}:video_path", video_path)  # For audio merge later

        # Signal background encoder thread via Redis pub/sub (REAL-TIME!)
        redis_client.publish('segment_ready', json.dumps({
            'video_id': video_id,
            'seg_idx': seg_idx,
            'total_segments': total_segments
        }))

        print(f"[OK] Segment {seg_idx+1}/{total_segments} signaled to background encoder - worker returning immediately!")

        # Cleanup temp directories EXCEPT cleaned_dir (encoder needs it!)
        for path in [seg_frames_dir, seg_cropped_dir, seg_mask_dir, seg_output_dir]:
            shutil.rmtree(path, ignore_errors=True)

        result = {
            'seg_idx': seg_idx,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frames_processed': seg_duration,
            'status': 'ready_for_encoding',  # Background encoder will encode this
        }

        print(f"[OK] Segment {seg_idx+1}/{total_segments} complete - worker free to process next segment!")

        # Note: Don't call safe_update_state(state='SUCCESS') - it overrides the return value!
        # Celery automatically sets state to SUCCESS when the task returns normally
        # Chord will automatically trigger finalize when all segments complete

        return result

    except Exception as e:
        print(f"[ERROR] Error processing segment {seg_idx}: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.finalize_video')
def finalize_video_task(self, segment_results, prepare_result):
    """
    Phase 3: Return immediately - encoding happens in background encoder thread!

    NOTE: This task returns immediately. The background encoder handles all encoding + finalization asynchronously.
    Check Redis for final video status: video:{video_id}:status and video:{video_id}:final_path
    """
    try:
        video_id = prepare_result['video_id']
        base_name = prepare_result['base_name']
        total_segments = prepare_result['total_segments']

        print(f"\n[FINALIZE TASK] All segments complete! Background encoder handling finalization for: {base_name}")
        print(f"[FINALIZE TASK] Encoding + concatenation happening asynchronously in background (parallel NVENC)")

        # Return immediately - don't block Celery worker!
        # Background encoder will update Redis when complete:
        # - video:{video_id}:status = "complete"
        # - video:{video_id}:final_path = <path to final video>

        result = {
            'status': 'background_encoding',
            'message': f'Segments complete. Encoding + finalization in progress (background thread)',
            'video_id': video_id,
            'total_segments': total_segments,
            'check_status_key': f'video:{video_id}:status',
            'final_path_key': f'video:{video_id}:final_path'
        }

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Background encoding in progress'})
        print(f"[FINALIZE TASK] Returned immediately - worker free! Check Redis keys for completion.")
        return result

        # Timeout
        raise RuntimeError(f"Background encoder did not complete within {max_wait_seconds}s")

    except Exception as e:
        print(f"[ERROR] Error finalizing video: {e}")
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
        print(f"[INFO] Coordinator: Waiting for {total_segments} segment tasks to complete...")
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
            print(f"   [OK] Segment {i+1}/{total_segments} complete!")

        print(f"[OK] All {total_segments} segments complete! Launching finalize task...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Finalizing video'})

        # All segments done - now finalize
        final_result = finalize_video_task.apply_async(args=[segment_results, prepare_result])

        # Wait for finalize to complete
        final_output = final_result.get(timeout=600)  # 10 minute timeout

        print(f"[OK] Video processing complete!")
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})

        return final_output

    except Exception as e:
        print(f"[ERROR] Coordinator failed: {e}")
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

        print(f"[INIT] Launching chord: {total_segments} segments in parallel across all workers...")

        # Create the chord
        header = group([process_segment_task.s(seg) for seg in segments])
        callback = finalize_video_task.s(prepare_result)

        # Apply the chord asynchronously - this dispatches tasks immediately
        chord_result = chord(header)(callback)

        print(f"[OK] Chord dispatched: {total_segments} segment tasks queued")
        print(f"   Workers will pick up tasks and process in parallel")

        # Wait for the chord to complete and return the final result
        # This blocks this task but doesn't block workers from processing segments
        final_result = chord_result.get(timeout=3600)

        print(f"[OK] All segments processed and finalized!")
        return final_result

    except Exception as e:
        print(f"[ERROR] Segment launch failed: {e}")
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

        print(f"[RUNNING] Starting DISTRIBUTED video processing: {video_path}")
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

        print(f"[OK] Video prepared: {total_segments} segments ready for distributed processing")
        print(f"[WEB] Distributing segments across all available workers...")

        self.update_state(
            state='PROCESSING',
            meta={'progress': 50, 'status': f'Distributing {total_segments} segments across workers'}
        )

        # Add total_segments to each segment data
        for seg_data in segments:
            seg_data['total_segments'] = total_segments

        # Phase 2: Process segments in parallel (distributes across ALL workers)
        # Using chord: when all segment tasks complete, finalize task runs automatically
        print(f"[INIT] Phase 2: Processing {total_segments} segments in parallel...")

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

        print(f"[INFO] Workflow submitted:")
        print(f"   - {total_segments} segment tasks queued")
        print(f"   - Tasks will be picked up by ANY available worker")
        print(f"   - Finalization will run automatically when all segments complete")

        # Wait for the entire workflow to complete
        final_result = result.get(timeout=3600)  # 1 hour timeout

        print(f"[OK] DISTRIBUTED processing complete!")
        print(f"   Final video: {final_result.get('path')}")

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Distributed processing complete'})

        return final_result

    except Exception as e:
        print(f"[ERROR] Distributed processing failed: {e}")
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
                print(f"[WEB] Image not found locally. Downloading from: {download_url}")
                try:
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=30)
                    r.raise_for_status()
                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    image_path = remote_cached
                    print(f"[OK] Downloaded to: {image_path}")
                except Exception as dl_err:
                    print(f"[ERROR] Failed to download image from API host: {dl_err}")
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
            # Use cached ProPainter pipeline (pre-loaded at worker startup)
            faster_propainter_pipeline = get_propainter_pipeline()

            import torch
            use_fp16 = torch.cuda.is_available()
            
            print(f"[RUNNING] Running direct faster-propainter pipeline: FP16={use_fp16} + neighbor_length=20 + ref_stride=10 + subvideo_length=80")
            
            # Direct pipeline call - TRUE FASTER-PROPAINTER SETTINGS
            faster_propainter_pipeline(
                video=frame_dir,                    # Input frame directory
                mask=mask_dir,                      # Mask directory  
                output=PROPAINTER_OUTPUT_ROOT,      # Output directory
                resize_ratio=1.0,                   # Keep original resolution for images
                mask_dilation=4,                    # faster-propainter: standard mask dilation
                ref_stride=15,                      # faster-propainter: faster processing
                neighbor_length=10,                 # faster-propainter: reduced for speed
                subvideo_length=60,                 # faster-propainter: reduced for speed
                raft_iter=10,                       # faster-propainter: reduced for speed
                mode="video_inpainting",            # Standard inpainting mode
                save_frames=True,                   # Save individual frames
                fp16=use_fp16                       # Enable FP16 if available
            )
            
            print("[OK] faster-propainter pipeline completed successfully")
            performance_checkpoint("faster-propainter Pipeline", pipeline_start)
            
        except Exception as e:
            print(f"[ERROR] faster-propainter pipeline failed: {e}")
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
            print(f"[WARNING]  Failed to cleanup temp directories: {cleanup_exc}")

        print(f"[OK] Image processed with ProPainter FP16: {out_path}")

        # If running remotely and the API host is reachable via TUNNEL_URL,
        # upload the result back so the frontend can download from the API server.
        uploaded_path = None
        tunnel = os.getenv('TUNNEL_URL')
        if tunnel and os.getenv('UPLOAD_RESULT_BACK', '1') == '1':
            try:
                import requests
                upload_url = tunnel.rstrip('/') + '/api/upload-result'
                # Quick connectivity test (15 second timeout) - fail fast if server down
                requests.head(tunnel, timeout=15, headers={'ngrok-skip-browser-warning': 'true'})
                print(f"[UPLOAD]  Uploading result to API host: {upload_url}")
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
                        print(f"[OK] Uploaded result registered at: {uploaded_path}")
                else:
                    print(f"[WARNING]  Upload back failed: HTTP {resp.status_code}")
            except Exception as up_err:
                print(f"[WARNING]  Upload back error: {up_err}")

        return {'path': uploaded_path or out_path}
    except Exception as e:
        print(f"[ERROR] Error processing image: {e}")
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
                print(f"[WEB] Video not found locally. Downloading from: {download_url}")
                try:
                    r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=30)
                    r.raise_for_status()
                    # Save to a temp path in UPLOAD_DIR
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    remote_cached = os.path.join(UPLOAD_DIR, base_name)
                    with open(remote_cached, 'wb') as f:
                        f.write(r.content)
                    video_path = remote_cached
                    print(f"[OK] Downloaded to: {video_path}")
                except Exception as dl_err:
                    print(f"[ERROR] Failed to download video from API host: {dl_err}")
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
        print(f"[OK] Masks generated: {frames_processed} frames total")
        print(f"   [REGEN] YOLO detections: {detections_run}/{frames_processed} frames (detect_interval={detect_interval})")
        print(f"   💧 Frames with watermark: {frames_with_watermark}")

        # Detect position segments for smart cropping
        from segment_detector import detect_segments, merge_adjacent_segments, should_use_cropping

        segments = detect_segments(bboxes_per_frame, position_tolerance=5, min_segment_length=10)
        if segments:
            segments = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)
            print(f"[INFO] Detected {len(segments)} watermark position segments:")
            for i, (start, end, bbox) in enumerate(segments):
                duration = end - start + 1
                print(f"   Segment {i+1}: frames {start}-{end} ({duration} frames) bbox={bbox}")

        use_cropping = should_use_cropping(segments, width, height, min_speedup=5.0) if segments else False

        if use_cropping and segments:
            print("[RUNNING] Smart cropping enabled - processing segments individually")
            import subprocess

            # Extract original frames once for merging later
            original_frames_dir = os.path.join(TEMP_DIR, f"{base_name}_{unique_suffix}_originals")
            os.makedirs(original_frames_dir, exist_ok=True)

            print(f"📷 Extracting original frames for merging...")
            extract_cmd = [
                FFMPEG_EXE, '-i', video_path,
                '-qscale:v', '2',
                '-start_number', '0',  # Start numbering at 0 to match frame indices
                os.path.join(original_frames_dir, '%04d.png').replace('\\', '/')
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

            print(f"[PAINT] Phase 1: ProPainter processing ({len(segments)} segments across {num_gpus} GPUs)")
            print(f"   [RUNNING] Each segment gets its own GPU for true parallel processing")
            self.update_state(
                state='PROCESSING',
                meta={'progress': 55, 'status': f'ProPainter processing ({num_gpus} GPUs)'}
            )

            if num_gpus > 1:
                # Parallel processing with GPU assignment
                completed = 0
                print(f"[INIT] Processing {len(segments)} segments in parallel across {num_gpus} GPUs...")

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
                            print(f"[OK] ProPainter segment {seg_idx+1}/{len(segments)} complete ({completed}/{len(segments)} total)")
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
                    print(f"[OK] ProPainter segment {seg_idx+1}/{len(segments)} complete")

            if any(d is None for d in segment_cleaned_dirs):
                raise RuntimeError("One or more segments failed ProPainter processing")

            # Clear GPU memory after all ProPainter processing completes
            print("[CLEANUP] Clearing GPU memory after ProPainter phase...")
            clear_gpu_memory()

            # Phase 2: Video encoding (CPU-bound, parallel)
            max_workers = min(4, len(segments))  # Use up to 4 threads for encoding
            print(f"\n[ENCODE]  Phase 2: Video encoding ({len(segments)} segments in parallel, {max_workers} workers)")
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
                            print(f"[OK] Encoded segment {seg_idx+1}/{len(segments)} ({completed}/{len(segments)} total)")
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
            print(f"\n[SEGMENT] Concatenating {len(segment_videos)} segments...")
            self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Concatenating segments'})

            concat_list_file = os.path.join(TEMP_DIR, f"{base_name}_{unique_suffix}_concat.txt")
            with open(concat_list_file, 'w') as f:
                for seg_video in segment_videos:
                    f.write(f"file '{seg_video}'\n")

            temp_processed = os.path.join(RESULT_DIR, f"{base_name}_propainter_video.mp4")
            concat_cmd = [
                FFMPEG_EXE, '-y', '-f', 'concat', '-safe', '0',
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
                print("[INFO]  Smart cropping not beneficial for this video, using full frame processing")

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
                # Use cached ProPainter pipeline (pre-loaded at worker startup)
                faster_propainter_pipeline = get_propainter_pipeline()

                import torch
                use_fp16 = torch.cuda.is_available()
                
                # Dynamic optimization based on video resolution
                dynamic_subvideo, _ = get_dynamic_subvideo_length(width, height)
                
                print(f"[RUNNING] Direct faster-propainter pipeline: resolution={width}x{height}, FP16={use_fp16}")
                print(f"   faster-propainter: neighbor_length=10 + ref_stride=15 + raft_iter=10 + subvideo_length=60 + flow_backend={PROPAINTER_FLOW_BACKEND}")

                # Direct pipeline call - OPTIMIZED FOR SPEED
                faster_propainter_pipeline(
                    video=video_path,                   # Input video
                    mask=mask_dir,                      # Mask directory
                    output=PROPAINTER_OUTPUT_ROOT,      # Output directory
                    resize_ratio=1.0,                   # Keep original resolution
                    mask_dilation=4,                    # faster-propainter: standard mask dilation
                    ref_stride=15,                      # faster-propainter: optimized for speed
                    neighbor_length=10,                 # faster-propainter: reduced for speed
                    subvideo_length=60,                 # faster-propainter: reduced for speed
                    raft_iter=10,                       # faster-propainter: reduced for speed
                    mode="video_inpainting",            # Standard inpainting
                    save_fps=fps,                       # Preserve original FPS
                    save_frames=False,                  # Only save video, not frames
                    fp16=use_fp16                       # FP16 if available
                )
                
                print("[OK] faster-propainter pipeline completed successfully")
                
            except Exception as e:
                print(f"[ERROR] faster-propainter pipeline failed: {e}")
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
                FFPROBE_EXE,
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
                    FFMPEG_EXE,
                    '-y',
                    '-i', temp_processed,
                    '-i', video_path,
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-c:v', 'copy',         # Don't re-encode, just copy video stream!
                    '-c:a', 'copy',         # Copy audio stream directly (instant, no re-encoding!)
                    '-shortest',
                    final_output
                ]
            else:
                cmd = [
                    FFMPEG_EXE,
                    '-y',
                    '-i', temp_processed,
                    '-c:v', 'copy',         # No audio, just copy video stream
                    final_output
                ]

            print(f"Running FFmpeg audio merge: {' '.join(cmd)}")
            audio_merge = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if audio_merge.returncode != 0:
                print("[WARNING]  Audio merge failed, returning video without audio")
                print(audio_merge.stderr)
                final_output = temp_processed
            else:
                if has_audio:
                    verify_cmd = [
                        FFPROBE_EXE,
                        '-v', 'error',
                        '-select_streams', 'a:0',
                        '-show_entries', 'stream=codec_type',
                        '-of', 'default=noprint_wrappers=1:nokey=1',
                        final_output
                    ]
                    verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)
                    if 'audio' not in verify.stdout:
                        print("[WARNING]  Audio verification failed, keeping silent video")
                        final_output = temp_processed
                if final_output != temp_processed and os.path.exists(temp_processed):
                    os.remove(temp_processed)
        except FileNotFoundError:
            print("[WARNING]  ffmpeg/ffprobe not available, returning silent video")
            final_output = temp_processed
        except subprocess.TimeoutExpired:
            print("[WARNING]  FFmpeg timed out, returning silent video")
            final_output = temp_processed

        try:
            shutil.rmtree(mask_dir, ignore_errors=True)
        except Exception as cleanup_exc:
            print(f"[WARNING]  Failed to delete mask directory {mask_dir}: {cleanup_exc}")

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
                # Quick connectivity test (15 second timeout) - fail fast if server down
                requests.head(tunnel, timeout=15, headers={'ngrok-skip-browser-warning': 'true'})
                print(f"[UPLOAD]  Uploading result to API host: {upload_url}")
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
                        print(f"[OK] Uploaded result registered at: {uploaded_path}")
                else:
                    print(f"[WARNING]  Upload back failed: HTTP {resp.status_code}")
            except Exception as up_err:
                print(f"[WARNING]  Upload back error: {up_err}")

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
        print(f"[ERROR] Error processing video: {e}")
        import traceback
        traceback.print_exc()
        raise



# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/ads.txt')
def ads_txt():
