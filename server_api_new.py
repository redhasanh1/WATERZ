"""
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

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return ('', 204)
    """Basic health endpoint for monitoring."""
    return jsonify({
        'status': 'ok',
        'message': 'Flask API server running - workers handle processing'
    })

# ----------------------------------------------------------------------------
# Authentication Routes (Google OAuth + Email/Password)
# ----------------------------------------------------------------------------

@app.route('/auth/google')
def auth_google():
    """Initiate Google OAuth flow."""
    if not AUTH_ENABLED or not GOOGLE_CLIENT_ID:
        return jsonify({'error': 'Google OAuth not configured'}), 503

    try:
        # Create flow instance
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [f"{request.host_url}auth/google/callback"]
                }
            },
            scopes=['openid', 'email', 'profile']
        )

        # Use the actual callback URL from request
        flow.redirect_uri = f"{request.host_url}auth/google/callback"

        # Generate authorization URL with state token (CSRF protection)
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='select_account'
        )

        # Store state in session for verification
        session['oauth_state'] = state

        return redirect(authorization_url)

    except Exception as e:
        print(f"[ERROR] Google OAuth initiation failed: {e}")
        return jsonify({'error': 'Failed to initiate Google login'}), 500


@app.route('/auth/google/callback')
def auth_google_callback():
    """Handle Google OAuth callback."""
    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication not enabled'}), 503

    try:
        # Verify state token (CSRF protection)
        state = session.get('oauth_state')
        if not state or state != request.args.get('state'):
            return jsonify({'error': 'Invalid state parameter'}), 400

        # Create flow instance with same config
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [f"{request.host_url}auth/google/callback"]
                }
            },
            scopes=['openid', 'email', 'profile'],
            state=state
        )

        flow.redirect_uri = f"{request.host_url}auth/google/callback"

        # Exchange authorization code for tokens
        flow.fetch_token(authorization_response=request.url)

        # Get user info from ID token
        credentials = flow.credentials
        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Extract user information
        google_id = id_info['sub']
        email = id_info['email']
        name = id_info.get('name', email.split('@')[0])

        # Store or update user in database
        with get_db() as conn:
            cur = conn.cursor()

            # Check if user exists
            cur.execute('SELECT id, credits FROM users WHERE google_id = %s', (google_id,))
            user = cur.fetchone()

            if user:
                user_id, credits = user
                print(f"[AUTH] Existing user logged in: {email}")
            else:
                # New user - give 5 free credits
                cur.execute(
                    'INSERT INTO users (google_id, email, name, credits) VALUES (%s, %s, %s, %s) RETURNING id',
                    (google_id, email, name, 5)
                )
                user_id = cur.fetchone()[0]
                credits = 5
                print(f"[AUTH] New user registered via Google: {email} (5 free credits)")

            # Create session
            session['user_id'] = user_id
            session['email'] = email
            session['name'] = name
            session.permanent = True

        # Redirect to main page
        return redirect('/')

    except Exception as e:
        print(f"[ERROR] Google OAuth callback failed: {e}")
        return jsonify({'error': 'Authentication failed'}), 500


@app.route('/api/auth/register', methods=['POST', 'OPTIONS'])
def auth_register():
    """Register new user with email/password."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication not enabled'}), 503

    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()

        # Validation
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400

        # Basic email validation
        if '@' not in email or '.' not in email.split('@')[1]:
            return jsonify({'error': 'Invalid email address'}), 400

        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Store user in database
        with get_db() as conn:
            cur = conn.cursor()

            # Check if email already exists
            cur.execute('SELECT id FROM users WHERE email = %s', (email,))
            if cur.fetchone():
                return jsonify({'error': 'Email already registered'}), 409

            # Create user with 5 free credits
            cur.execute(
                'INSERT INTO users (email, password_hash, name, credits) VALUES (%s, %s, %s, %s) RETURNING id',
                (email, password_hash, name or email.split('@')[0], 5)
            )
            user_id = cur.fetchone()[0]

            # Create session
            session['user_id'] = user_id
            session['email'] = email
            session['name'] = name or email.split('@')[0]
            session.permanent = True

            print(f"[AUTH] New user registered: {email} (5 free credits)")

            return jsonify({
                'status': 'success',
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': session['name'],
                    'credits': 5
                }
            })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Registration failed: {e}")
        print(f"[ERROR] Full traceback:\n{error_details}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500


@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def auth_login():
    """Login with email/password."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication not enabled'}), 503

    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        with get_db() as conn:
            cur = conn.cursor()

            # Get user
            cur.execute('SELECT id, password_hash, name, credits FROM users WHERE email = %s', (email,))
            user = cur.fetchone()

            if not user:
                return jsonify({'error': 'Invalid email or password'}), 401

            user_id, password_hash, name, credits = user

            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8')):
                return jsonify({'error': 'Invalid email or password'}), 401

            # Create session
            session['user_id'] = user_id
            session['email'] = email
            session['name'] = name
            session.permanent = True

            print(f"[AUTH] User logged in: {email}")

            return jsonify({
                'status': 'success',
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': name,
                    'credits': credits
                }
            })

    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Check if user is logged in and get user info."""
    if not AUTH_ENABLED:
        return jsonify({'authenticated': False})

    try:
        user_id = session.get('user_id')

        if not user_id:
            return jsonify({'authenticated': False})

        # Get current user data from database
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('SELECT email, name, credits FROM users WHERE id = %s', (user_id,))
            user = cur.fetchone()

            if not user:
                session.clear()
                return jsonify({'authenticated': False})

            email, name, credits = user

            return jsonify({
                'authenticated': True,
                'user': {
                    'id': user_id,
                    'email': email,
                    'name': name,
                    'credits': credits
                }
            })

    except Exception as e:
        print(f"[ERROR] Auth status check failed: {e}")
        return jsonify({'authenticated': False})


@app.route('/api/auth/logout', methods=['POST', 'OPTIONS'])
def auth_logout():
    """Logout user."""
    if request.method == 'OPTIONS':
        return ('', 204)

    session.clear()
    return jsonify({'status': 'success'})


# ----------------------------------------------------------------------------
# User Profile Routes
# ----------------------------------------------------------------------------

@app.route('/api/user/profile', methods=['GET', 'OPTIONS'])
def get_user_profile():
    """Get user profile information."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication disabled'}), 500

    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'signin_required'}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, email, name, credits, created_at, google_id
            FROM users
            WHERE id = %s
        ''', (user_id,))

        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({
            'id': user[0],
            'email': user[1],
            'name': user[2],
            'credits': user[3],
            'created_at': user[4].isoformat() if user[4] else None,
            'has_google_account': user[5] is not None
        })

    except Exception as e:
        print(f"Error getting profile: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to get profile'}), 500


@app.route('/api/user/profile', methods=['PUT', 'OPTIONS'])
def update_user_profile():
    """Update user profile (name, email)."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication disabled'}), 500

    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'signin_required'}), 401

    try:
        data = request.get_json()
        name = data.get('name', '').strip()

        if not name:
            return jsonify({'error': 'Name is required'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE users
            SET name = %s
            WHERE id = %s
            RETURNING id, email, name, credits
        ''', (name, user_id))

        user = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({
            'id': user[0],
            'email': user[1],
            'name': user[2],
            'credits': user[3]
        })

    except Exception as e:
        print(f"Error updating profile: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to update profile'}), 500


@app.route('/api/user/change-password', methods=['POST', 'OPTIONS'])
def change_password():
    """Change user password."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication disabled'}), 500

    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'signin_required'}), 401

    try:
        data = request.get_json()
        current_password = data.get('currentPassword', '')
        new_password = data.get('newPassword', '')

        if not current_password or not new_password:
            return jsonify({'error': 'Current and new passwords are required'}), 400

        if len(new_password) < 8:
            return jsonify({'error': 'New password must be at least 8 characters'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get current password hash
        cursor.execute('''
            SELECT password_hash, google_id
            FROM users
            WHERE id = %s
        ''', (user_id,))

        user = cursor.fetchone()

        if not user:
            cursor.close()
            conn.close()
            return jsonify({'error': 'User not found'}), 404

        # Check if user has Google account (no password)
        if user[1] is not None and user[0] is None:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Google accounts cannot set passwords. Please use Google login.'}), 400

        # Verify current password
        if not user[0] or not bcrypt.checkpw(current_password.encode('utf-8'), user[0].encode('utf-8')):
            cursor.close()
            conn.close()
            return jsonify({'error': 'Current password is incorrect'}), 401

        # Hash new password
        new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Update password
        cursor.execute('''
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
        ''', (new_password_hash, user_id))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'status': 'success', 'message': 'Password changed successfully'})

    except Exception as e:
        print(f"Error changing password: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to change password'}), 500


# ----------------------------------------------------------------------------
# Password Reset Routes
# ----------------------------------------------------------------------------

@app.route('/api/auth/forgot-password', methods=['POST', 'OPTIONS'])
def forgot_password():
    """Request password reset email."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication disabled'}), 500

    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()

        if not email:
            return jsonify({'error': 'Email is required'}), 400

        with get_db() as conn:
            cursor = conn.cursor()

            # Check if user exists
            cursor.execute('SELECT id, name FROM users WHERE email = %s', (email,))
            user = cursor.fetchone()

            # Always return success to prevent email enumeration
            # But only send email if user exists
            if user:
                import secrets
                import datetime

                # Generate reset token
                token = secrets.token_urlsafe(32)
                expires_at = datetime.datetime.now() + datetime.timedelta(hours=1)

                # Store token in database
                cursor.execute('''
                    INSERT INTO password_reset_tokens (user_id, token, expires_at)
                    VALUES (%s, %s, %s)
                ''', (user[0], token, expires_at))

                # Generate reset URL and send email
                reset_url = f"{request.host_url}reset-password.html?token={token}"

                try:
                    send_reset_email(email, reset_url)
                except Exception as mail_err:
                    print(f"[WARNING] Email failed for {email}: {mail_err}")
                    # Still return success to prevent email enumeration

            cursor.close()

        return jsonify({
            'status': 'success',
            'message': 'If an account exists with that email, a password reset link has been sent.'
        })

    except Exception as e:
        print(f"Error in forgot password: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to process request'}), 500


@app.route('/api/auth/reset-password', methods=['POST', 'OPTIONS'])
def reset_password():
    """Reset password using token."""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not AUTH_ENABLED:
        return jsonify({'error': 'Authentication disabled'}), 500

    try:
        data = request.get_json()
        token = data.get('token', '')
        new_password = data.get('newPassword', '')

        if not token or not new_password:
            return jsonify({'error': 'Token and new password are required'}), 400

        if len(new_password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400

        with get_db() as conn:
            cursor = conn.cursor()

            # Find valid token
            cursor.execute('''
                SELECT user_id, expires_at
                FROM password_reset_tokens
                WHERE token = %s AND used = FALSE
            ''', (token,))

            reset_token = cursor.fetchone()

            if not reset_token:
                cursor.close()
                return jsonify({'error': 'Invalid or expired reset token'}), 400

            # Check if token expired
            import datetime
            if reset_token[1] < datetime.datetime.now():
                cursor.close()
                return jsonify({'error': 'Reset token has expired'}), 400

            # Hash new password
            password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Update password
            cursor.execute('''
                UPDATE users
                SET password_hash = %s
                WHERE id = %s
            ''', (password_hash, reset_token[0]))

            # Mark token as used
            cursor.execute('''
                UPDATE password_reset_tokens
                SET used = TRUE
                WHERE token = %s
            ''', (token,))

            cursor.close()

        return jsonify({'status': 'success', 'message': 'Password reset successfully'})

    except Exception as e:
        print(f"Error resetting password: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to reset password'}), 500


# ----------------------------------------------------------------------------
# End Authentication Routes
# ----------------------------------------------------------------------------

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
    encoder_name = ENCODER_CONFIG['name'] if ENCODER_CONFIG else 'default'
    print(f"   [ENCODE]  Encoding segment {seg_label} to video ({encoder_name})...")

    try:
        # Use detected encoder (NVENC if available, libx264 CPU fallback)
        encode_cmd = [
            FFMPEG_EXE, '-y',
            '-framerate', str(fps),
            '-start_number', '0',  # Start from 0000.png
            '-i', os.path.join(seg_cleaned_dir, '%04d.png').replace('\\', '/'),
            '-c:v', ENCODER_CONFIG['codec'],
            '-preset', ENCODER_CONFIG['preset'],
            '-b:v', '8M',  # Increased bitrate for better quality at higher speed
            '-bufsize', '16M',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'main',
            seg_video_path
        ]
        result = subprocess.run(encode_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Fallback to slower preset if fast preset fails
            print(f"   [WARNING]  {ENCODER_CONFIG['preset']} preset failed, trying {ENCODER_CONFIG['fallback_preset']}...")
            print(f"   [DEBUG] Error: {result.stderr[:500]}")  # Show first 500 chars of error
            encode_cmd[encode_cmd.index('-preset') + 1] = ENCODER_CONFIG['fallback_preset']
            result = subprocess.run(encode_cmd, capture_output=True, text=True, check=True)

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

