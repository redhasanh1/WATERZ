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

    # 🔥 UPLOAD TO RAILWAY: If running on local worker, upload result to API server
    uploaded_path = None
    tunnel = os.getenv('TUNNEL_URL') or os.getenv('API_BASE_URL')
    upload_enabled = os.getenv('UPLOAD_RESULT_BACK', '1')

    print(f"[FINALIZE] Upload config - TUNNEL_URL: {'SET' if os.getenv('TUNNEL_URL') else 'NOT SET'}, API_BASE_URL: {'SET' if os.getenv('API_BASE_URL') else 'NOT SET'}, UPLOAD_RESULT_BACK: {upload_enabled}")

    if not tunnel:
        print(f"[FINALIZE] ⚠️  Skipping Railway upload - TUNNEL_URL/API_BASE_URL not set")
        print(f"[FINALIZE] 💡 Set TUNNEL_URL=https://your-railway-app.railway.app to enable auto-upload")
    elif upload_enabled != '1':
        print(f"[FINALIZE] ⚠️  Skipping Railway upload - UPLOAD_RESULT_BACK={upload_enabled} (set to '1' to enable)")

    if tunnel and upload_enabled == '1':
        try:
            import requests
            upload_url = tunnel.rstrip('/') + '/api/upload-result'
            print(f"[FINALIZE] 📤 Uploading result to Railway: {upload_url}")

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
                    print(f"[FINALIZE] ✅ Result uploaded to Railway: {uploaded_path}")
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

    # 🔥 FIX: Update distributed tracking to mark all segments complete
    # Status endpoint checks segments:{video_id} to see progress
    # Without this, frontend shows "Segment 0/X complete" forever
    tracking_key = f"segments:{video_id}"
    total_segments_bytes = redis_client.get(f"{tracking_key}:total")
    if total_segments_bytes:
        redis_client.set(tracking_key, int(total_segments_bytes))  # Mark all segments complete
        print(f"[FINALIZE] ✅ Marked all {int(total_segments_bytes)} segments complete in Redis tracking")

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
            print(f"   ⚡ Skipping disk-based cropping - using in-memory pipeline (saves ~600ms!)")

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
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'No watermark - encoding original'})

            # Copy original frames to cleaned dir (no processing needed)
            # 🔥 SHARED BUFFER: Use global frame indices
            if using_memory_pipeline:
                # Write from memory (only core segment, skip padding)
                # 🔥 TEMPORAL CONTEXT FIX: Calculate padding offset
                padding_offset = start_frame - padded_start
                core_frames = segment_frames_memory[padding_offset:padding_offset + seg_duration]
                print(f"   💾 Writing {len(core_frames)} core segment frames from memory to shared buffer (skipping {padding_offset} padding frames)...")
                for local_idx, frame in enumerate(core_frames):
                    # Use GLOBAL frame index for shared buffer
                    global_frame_idx = start_frame + local_idx
                    frame_file = f"{global_frame_idx:04d}.png"
                    dst = os.path.join(shared_cleaned_dir, frame_file)

                    # Only write if not already written by another segment
                    if not os.path.exists(dst):
                        cv2.imwrite(dst, frame)
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

        else:
            # Run ProPainter on this segment - watermark detected!
            print(f"   [PAINT] Running ProPainter on {frames_with_watermark} watermarked frames...")
            self.update_state(state='PROCESSING', meta={'progress': 50, 'status': f'Running ProPainter'})

            try:
                # Use cached ProPainter pipeline (pre-loaded at worker startup)
                faster_propainter_pipeline = get_propainter_pipeline()

                import torch
                use_fp16 = torch.cuda.is_available()

                # [INIT] EXTREME SPEED: Use already-loaded memory frames (skip FRAME_CACHE re-access!)
                frames_array = None
                masks_array = None

                if using_memory_pipeline:
                    print(f"   ⚡ Cropping {len(segment_frames_memory)} frames/masks in memory (ZERO disk I/O!)")
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
            self.update_state(state='PROCESSING', meta={'progress': 80, 'status': f'Merging results'})

            seg_propainter_frames = os.path.join(seg_output_dir, os.path.basename(seg_cropped_dir), 'frames')
            if not os.path.exists(seg_propainter_frames):
                raise RuntimeError(f"ProPainter output not found for segment {seg_idx}")

            # Load all frames into memory first (faster than disk I/O in loop)
            # [INIT] EXTREME SPEED: Use in-memory frames if available (no disk reads!)
            original_frames = []
            original_masks = []  # 🔥 MASK COMPOSITING: Need masks for alpha blending
            cleaned_frames = []

            # 🔥 TEMPORAL CONTEXT FIX: Calculate padding offset
            # ProPainter processed frames WITH padding, we only need core segment frames
            padding_offset = start_frame - padded_start  # How many padding frames before core segment
            print(f"   [CONTEXT] ProPainter processed {len(segment_frames_memory)} frames (padding offset: {padding_offset})")

            if using_memory_pipeline:
                # Use already-loaded memory frames (ZERO disk I/O!)
                # Extract ONLY core segment frames (skip padding)
                print(f"   ⚡ Extracting core segment frames + masks from memory (frames {padding_offset} to {padding_offset + seg_duration})...")
                original_frames = segment_frames_memory[padding_offset:padding_offset + seg_duration]
                original_masks = segment_masks_memory[padding_offset:padding_offset + seg_duration]
            else:
                # Disk-based fallback
                print(f"   [MASKS] Loading {seg_duration} original frames + masks from disk...")
                for frame_idx in range(seg_duration):
                    frame_file = f"{frame_idx:04d}.png"
                    orig = cv2.imread(os.path.join(seg_frames_dir, frame_file))
                    original_frames.append(orig)
                    # Load corresponding mask
                    mask = cv2.imread(os.path.join(seg_mask_dir, frame_file), cv2.IMREAD_GRAYSCALE)
                    original_masks.append(mask)

            # Load cleaned frames from ProPainter output (skip padding frames)
            print(f"   [OUTPUT] Extracting core segment from ProPainter output (frames {padding_offset} to {padding_offset + seg_duration})...")
            for frame_idx in range(padding_offset, padding_offset + seg_duration):
                frame_file = f"{frame_idx:04d}.png"
                clean = cv2.imread(os.path.join(seg_propainter_frames, frame_file))
                cleaned_frames.append(clean)

            # 🔥 SHARED BUFFER MERGE: Load existing frame state, apply this segment's edit, save back
            # This allows multiple segments to cooperatively edit the same frames
            print(f"   🔗 Merging to shared frame buffer with mask-based alpha compositing...")
            for local_idx, (original, cleaned_crop, segment_mask) in enumerate(zip(original_frames, cleaned_frames, original_masks)):
                # Use GLOBAL frame index for shared buffer
                global_frame_idx = start_frame + local_idx
                frame_file = f"{global_frame_idx:04d}.png"
                shared_frame_path = os.path.join(shared_cleaned_dir, frame_file)

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

                # 🔥 MASK COMPOSITING: Use mask to blend only the inpainted region
                # This preserves other segments' work in non-masked areas
                if cleaned_crop is not None and result_frame is not None and segment_mask is not None:
                    # Crop mask to ROI
                    cropped_mask = segment_mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

                    # Convert mask to 3-channel float [0, 1] for alpha blending
                    if len(cropped_mask.shape) == 2:
                        # Grayscale mask - convert to 3 channels
                        mask_3ch = cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
                    else:
                        mask_3ch = cropped_mask.astype(float) / 255.0

                    # Alpha composite: blend cleaned region using mask as alpha
                    roi = result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].astype(float)
                    cleaned_crop_float = cleaned_crop.astype(float)
                    blended = (cleaned_crop_float * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

                    # Paste blended result back
                    result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = blended

                # Save back to shared buffer
                cv2.imwrite(shared_frame_path, result_frame)

            print(f"   [OK] Merged {len(cleaned_frames)} frames to shared buffer: {shared_cleaned_dir}")

        # ⚡ PARALLEL ENCODING: Encode segment MP4 immediately (Blackwell NVENC!)
        print(f"   [OK] Cleaned frames ready in: {seg_cleaned_dir}")
        # ⚡ BACKGROUND ENCODING OPTIMIZATION: Signal encoder thread instead of blocking!
        # Worker returns immediately and starts next segment while encoding happens in background
        self.update_state(state='PROCESSING', meta={'progress': 85, 'status': f'Segment complete - signaling encoder'})

        # Count cleaned frames
        cleaned_frame_count = len([f for f in os.listdir(seg_cleaned_dir) if f.endswith('.png')])
        if cleaned_frame_count == 0:
            raise RuntimeError(f"No cleaned frames found in {seg_cleaned_dir}")

        print(f"   [OK] {cleaned_frame_count} cleaned frames ready - signaling background encoder!")

        # Store segment metadata in Redis for background encoder
        redis_client = celery.backend.client
        segment_key = f"video:{video_id}:segment:{seg_idx}"
        redis_client.hset(segment_key, 'cleaned_dir', seg_cleaned_dir)
        redis_client.hset(segment_key, 'fps', str(fps))
        redis_client.hset(segment_key, 'frame_count', str(cleaned_frame_count))
        redis_client.hset(segment_key, 'base_name', base_name)
        redis_client.hset(segment_key, 'status', 'ready_for_encoding')
        # 🔥 SHARED BUFFER: Store frame range for encoder
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

        # Note: Don't call self.update_state(state='SUCCESS') - it overrides the return value!
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
    return send_file(os.path.join(app.static_folder, 'index.html'))

@app.route('/index.html')
def index_html():
    """Serve index.html explicitly"""
    return send_file(os.path.join(app.static_folder, 'index.html'))


@app.route('/login.html')
def login_page():
    """Serve login page"""
    return send_file('web/login.html')


@app.route('/web/<path:path>')
def serve_web(path):
    """Serve static files from /web/ prefix"""
    return send_file(f'web/{path}')


# Serve static files from root (for Railway deployment)
@app.route('/config.js')
def serve_config():
    return send_file(os.path.join(app.static_folder, 'config.js'), mimetype='application/javascript')

@app.route('/js/<path:path>')
def serve_js(path):
    return send_file(os.path.join(app.static_folder, 'js', path), mimetype='application/javascript')

@app.route('/css/<path:path>')
def serve_css(path):
    return send_file(os.path.join(app.static_folder, 'css', path), mimetype='text/css')

@app.route('/emblem.png')
def serve_emblem():
    return send_file(os.path.join(app.static_folder, 'emblem.png'), mimetype='image/png')

@app.route('/demos/<path:path>')
def serve_demos(path):
    return send_file(os.path.join(app.static_folder, 'demos', path))


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
        print(f"[ERROR] Error queuing task: {e}")
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

            print(f"[INFO] Distributed task status - {task_id}: {completed}/{total} segments complete")

            if completed < total:
                # Still processing segments
                progress = int((completed / total) * 90)  # 0-90%
                return jsonify({
                    'state': 'PROCESSING',
                    'progress': f'Processing segments: {completed}/{total}',
                    'info': {'progress': progress, 'status': f'Segment {completed}/{total} complete'}
                })
            else:
                # All segments done - check if finalization/encoding complete
                try:
                    redis_client = celery.backend.client
                    encoding_status = redis_client.get(f"video:{video_id}:status")

                    if encoding_status and encoding_status.decode() == "complete":
                        # Encoding is DONE! Get the final path
                        final_path_raw = redis_client.get(f"video:{video_id}:final_path")
                        if final_path_raw:
                            final_path = final_path_raw.decode()
                            # Check if path is web path or local path
                            if final_path.startswith('/results/'):
                                result_url = final_path
                            else:
                                filename = os.path.basename(final_path)
                                result_url = f'/results/{filename}'

                            print(f"[STATUS] ✅ Encoding complete for {video_id}! Returning: {result_url}")
                            return jsonify({
                                'state': 'SUCCESS',
                                'result': {'result_url': result_url},
                                'metadata': {'total_segments': total}
                            })

                    # Encoding still in progress
                    print(f"[STATUS] Segments complete ({completed}/{total}), encoding in progress for {video_id}")
                    return jsonify({
                        'state': 'PROCESSING',
                        'progress': 'Finalizing video (encoding in progress)...',
                        'info': {'progress': 95, 'status': 'Video encoding in background (almost done!)'}
                    })

                except Exception as e:
                    print(f"[ERROR] Failed to check finalization status: {e}")
                    # Fallback - keep showing encoding in progress
                    return jsonify({
                        'state': 'PROCESSING',
                        'progress': 'Finalizing video...',
                        'info': {'progress': 95, 'status': 'Background encoding'}
                    })

        # Regular Celery task handling
        from celery.result import AsyncResult

        task = AsyncResult(task_id, app=celery)

        # Always log for debugging stuck "Waiting in queue" issue
        print(f"[INFO] Status check - Task: {task_id}, State: {task.state}, Info: {task.info}")

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
            print(f"[POLL] DEBUG - task.result type: {type(result_data)}, value: {result_data}")

            if isinstance(result_data, dict):
                # Check if this is a finalize task with background encoding
                if result_data.get('status') == 'background_encoding':
                    video_id = result_data.get('video_id')
                    print(f"[POLL] Background encoding in progress for video {video_id}, checking Redis...")

                    # Check if encoding is complete in Redis
                    try:
                        redis_client = celery.backend.client
                        encoding_status = redis_client.get(f"video:{video_id}:status")

                        if encoding_status and encoding_status.decode() == "complete":
                            # Encoding is done! Get the final path
                            final_path_raw = redis_client.get(f"video:{video_id}:final_path")
                            if final_path_raw:
                                final_path = final_path_raw.decode()
                                # Check if path is already a web path (from Railway upload)
                                if final_path.startswith('/results/'):
                                    result_url = final_path
                                else:
                                    # Local path - extract filename
                                    filename = os.path.basename(final_path)
                                    result_url = f'/results/{filename}'
                                print(f"[POLL] Background encoding COMPLETE for {video_id}! Returning result_url: {result_url}")
                                response['result'] = {
                                    'result_url': result_url
                                }
                                if 'metadata' in result_data:
                                    response['metadata'] = result_data['metadata']
                                return jsonify(response)  # Return SUCCESS with result_url
                        else:
                            # Encoding still in progress - keep frontend polling
                            print(f"[POLL] Background encoding not complete yet for {video_id}, status: {encoding_status}")
                            response['state'] = 'PROCESSING'
                            response['progress'] = result_data.get('message', 'Encoding in progress...')
                            response['info'] = {
                                'progress': 95,
                                'status': 'Video encoding in background (almost done!)'
                            }
                            return jsonify(response)  # Keep polling
                    except Exception as e:
                        print(f"[ERROR] Failed to check Redis for background encoding status: {e}")
                        # Fall through to normal error handling
                        response['state'] = 'PROCESSING'
                        response['progress'] = 'Encoding in progress...'
                        response['info'] = {'progress': 95, 'status': 'Background encoding'}
                        return jsonify(response)

                # Check if this is a prepare task that returned chord_id
                if 'chord_id' in result_data:
                    chord_id = result_data['chord_id']
                    print(f"[RETRY] Prepare task complete, switching to chord tracking: {chord_id}")
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
                    print(f"[ERROR] Task {task_id} SUCCESS but no path in result: {result_data}")
                    return jsonify({'error': 'Invalid result format - missing path'}), 500
            else:
                result_path = result_data

            # Final safety check for None result_path
            if not result_path:
                print(f"[ERROR] Task {task_id} has None result_path")
                return jsonify({'error': 'Invalid result - path is None'}), 500

            filename = os.path.basename(result_path)
            response['result'] = {
                'result_url': f'/results/{filename}'
            }
            if isinstance(result_data, dict) and 'metadata' in result_data:
                response['metadata'] = result_data['metadata']
        elif task.state == 'FAILURE':
            response['error'] = str(task.info)
            print(f"[ERROR] Task failed: {task.info}")

        return jsonify(response)
    except Exception as e:
        print(f"[ERROR] Status endpoint error for task {task_id}: {e}")
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
        print(f"[ERROR] Error retrieving result: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download-from-url', methods=['POST', 'OPTIONS'])
@require_auth
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
            print("[RUNNING] Launching browser for video download...")
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

            print("[POLL] Extracting video URL...")

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
                    print(f"[WARNING]  Error checking video element: {e}")

            if video_urls:
                video_src = html.unescape(video_urls[0])
                print(f"[OK] Found video URL: {video_src}")

                # Download the video
                import requests
                response = requests.get(video_src, stream=True, timeout=300)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                browser.close()
                print(f"[OK] Video downloaded: {output_path}")

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
        print(f"[ERROR] Download error: {e}")
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
        print(f"[POLL] Looking for cookies at: {cookies_file}")
        print(f"[POLL] SCRIPT_DIR is: {SCRIPT_DIR}")

        with sync_playwright() as p:
            print("[RUNNING] Launching browser for Sora download...")
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
                print("[OK] Cookies loaded!")
            else:
                print(f"[WARNING]  No cookies found at: {cookies_file}")
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

            print("[POLL] Extracting video URL from page content...")

            video_src = None

            # Try to find video URLs in page content (works for Sora)
            content = page.content()
            import re
            video_urls = re.findall(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content)

            if video_urls:
                import html
                video_src = html.unescape(video_urls[0])  # Decode &amp; to &
                print(f"[OK] Found video URL in page content: {video_src}")
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
                print(f"[OK] Found video source: {video_src}")

                # Make absolute URL if relative
                if video_src.startswith('//'):
                    video_src = 'https:' + video_src
                elif video_src.startswith('/'):
                    from urllib.parse import urljoin
                    video_src = urljoin(url, video_src)

                print(f"[DOWNLOAD]  Downloading video...")

                # Download using Playwright's request context with retry logic
                max_retries = 3
                retry_delay = 3

                for retry in range(max_retries):
                    try:
                        if retry > 0:
                            print(f"[RETRY] Retry attempt {retry}/{max_retries-1}...")
                            time.sleep(retry_delay)
                        else:
                            # Small delay before first attempt to avoid rate limiting
                            time.sleep(1)

                        response = page.request.get(video_src)

                        if response.ok:
                            with open(output_path, 'wb') as f:
                                f.write(response.body())

                            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                            print(f"[OK] Downloaded successfully! Size: {file_size:.2f} MB")

                            browser.close()

                            return jsonify({
                                'status': 'success',
                                'task_id': task_id,
                                'video_url': f'/uploads/{task_id}.mp4'
                            })
                        else:
                            print(f"[WARNING]  Download failed: HTTP {response.status}")
                            if retry == max_retries - 1:
                                browser.close()
                                return jsonify({
                                    'status': 'error',
                                    'message': f'Download failed after {max_retries} attempts: HTTP {response.status}'
                                }), 500
                    except Exception as download_error:
                        print(f"[WARNING]  Download error: {download_error}")
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
@require_auth
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

        print(f"[OK] File uploaded: {file_path}")

        return jsonify({
            'status': 'success',
            'task_id': task_id
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/extract-frames/<task_id>', methods=['GET', 'OPTIONS'])
def extract_frames(task_id):
    """
    Extract frame thumbnails from uploaded video for timeline UI

    Returns: { "status": "success", "frames": [{frame_number, timestamp, thumbnail_url}], "total_frames", "fps", "duration" }
    """
    if request.method == 'OPTIONS':
        return ('', 204)

    try:
        # Find the uploaded video
        video_path = None
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
            if os.path.exists(test_path):
                video_path = test_path
                break

        if not video_path:
            return jsonify({'status': 'error', 'message': 'Video not found'}), 404

        # Check if frames are already cached
        cache_key = f'frames_{task_id}'
        with FRAME_CACHE_LOCK:
            if cache_key in FRAME_CACHE:
                return jsonify(FRAME_CACHE[cache_key])

        # Get video metadata using ffprobe
        probe_cmd = [
            FFPROBE_EXE, '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
            '-of', 'json',
            video_path
        ]

        import subprocess
        import json as json_module
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        probe_data = json_module.loads(probe_result.stdout)

        # Extract metadata
        stream = probe_data['streams'][0]
        fps_str = stream.get('r_frame_rate', '30/1')
        fps_num, fps_den = map(int, fps_str.split('/'))
        fps = fps_num / fps_den if fps_den != 0 else 30

        duration = float(stream.get('duration', 0))
        total_frames = int(stream.get('nb_frames', int(duration * fps)))

        # Extract thumbnails at 1-second intervals (or max 60 thumbnails for long videos)
        max_thumbnails = 60
        interval_seconds = max(1, int(duration / max_thumbnails))

        # Create thumbnails directory for this task
        thumbnails_dir = os.path.join(CACHE_DIR, 'thumbnails', task_id)
        os.makedirs(thumbnails_dir, exist_ok=True)

        frames_data = []

        # Extract thumbnails using ffmpeg
        for i in range(0, int(duration), interval_seconds):
            thumbnail_path = os.path.join(thumbnails_dir, f'frame_{i:04d}.jpg')

            # Skip if already exists
            if not os.path.exists(thumbnail_path):
                ffmpeg_cmd = [
                    FFMPEG_EXE, '-y',
                    '-ss', str(i),  # Seek to timestamp
                    '-i', video_path,
                    '-vframes', '1',  # Extract 1 frame
                    '-vf', 'scale=120:-1',  # Scale to 120px width (maintain aspect ratio)
                    '-q:v', '5',  # Quality (lower = better, 2-5 is good)
                    thumbnail_path
                ]
                subprocess.run(ffmpeg_cmd, capture_output=True, timeout=5)

            if os.path.exists(thumbnail_path):
                frame_number = int(i * fps)
                frames_data.append({
                    'frame_number': frame_number,
                    'timestamp': i,
                    'thumbnail_url': f'/api/thumbnail/{task_id}/frame_{i:04d}.jpg'
                })

        result = {
            'status': 'success',
            'frames': frames_data,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration
        }

        # Cache the result
        with FRAME_CACHE_LOCK:
            FRAME_CACHE[cache_key] = result

        return jsonify(result)

    except subprocess.TimeoutExpired:
        return jsonify({'status': 'error', 'message': 'Frame extraction timeout'}), 500
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/thumbnail/<task_id>/<filename>', methods=['GET'])
def serve_thumbnail(task_id, filename):
    """
    Serve thumbnail images for timeline
    """
    try:
        thumbnails_dir = os.path.join(CACHE_DIR, 'thumbnails', task_id)
        thumbnail_path = os.path.join(thumbnails_dir, filename)

        if not os.path.exists(thumbnail_path):
            return jsonify({'status': 'error', 'message': 'Thumbnail not found'}), 404

        from flask import send_file
        return send_file(thumbnail_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/process', methods=['POST', 'OPTIONS'])
@require_auth
@require_credits(min_credits=1)
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
                # prepare_video will use cached video if broadcast succeeded
                result = prepare_video_task.apply_async(
                    args=[video_path],
                    kwargs={'api_base': base, 'temp_base': base}
                )
            print(f"[OK] Task queued with ID: {result.id}")

            # Deduct 1 credit from user's balance after successful task creation
            user_id = session.get('user_id')
            if user_id and AUTH_ENABLED:
                try:
                    with get_db() as conn:
                        cur = conn.cursor()
                        cur.execute(
                            'UPDATE users SET credits = credits - 1 WHERE id = %s AND credits >= 1 RETURNING credits',
                            (user_id,)
                        )
                        deduct_result = cur.fetchone()
                        if deduct_result:
                            new_balance = deduct_result[0]
                            print(f"[CREDITS] User {user_id} processed video. Task: {result.id}, New balance: {new_balance}")
                        else:
                            print(f"[WARNING] Credit deduction failed for user {user_id} - insufficient credits")
                except Exception as e:
                    print(f"[ERROR] Failed to deduct credit: {e}")
                    # Don't fail the request if credit deduction fails - task already queued

            return jsonify({'status': 'success', 'task_id': result.id})

        except Exception as e:
            print(f"[ERROR] Failed to queue task: {e}")
            import traceback; traceback.print_exc()
            return jsonify({'status': 'error', 'message': f'Failed to connect to Redis: {str(e)}'}), 500

    except Exception as e:
        print(f"[ERROR] Process endpoint error: {e}")
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


@app.route('/cool.mp4')
def serve_cool_video():
    """Serve showcase video (cool.mp4)"""
    video_path = os.path.join(STATIC_VIDEOS_DIR, 'cool.mp4')

    if not os.path.exists(video_path):
        return jsonify({'error': 'Showcase video not found'}), 404

    return send_file(video_path, mimetype='video/mp4')


@app.route('/s2.mp4')
def serve_s2_video():
    """Serve s2 before video"""
    video_path = os.path.join(STATIC_VIDEOS_DIR, 's2.mp4')

    if not os.path.exists(video_path):
        return jsonify({'error': 's2 video not found'}), 404

    return send_file(video_path, mimetype='video/mp4')


@app.route('/s2removed.mp4')
def serve_s2removed_video():
    """Serve s2removed after video"""
    video_path = os.path.join(STATIC_VIDEOS_DIR, 's2removed.mp4')

    if not os.path.exists(video_path):
        return jsonify({'error': 's2removed video not found'}), 404

    return send_file(video_path, mimetype='video/mp4')


@app.route('/training/<filename>')
def serve_training_video(filename):
    """Serve training videos from volume"""
    # Sanitize filename to prevent path traversal
    filename = sanitize_filename(filename)

    # Only allow .mp4 files
    if not filename.endswith('.mp4'):
        return jsonify({'error': 'Invalid file type'}), 400

    video_path = os.path.join(TRAINING_VIDEOS_DIR, filename)

    # Verify file exists and is within training directory
    if not os.path.exists(video_path) or not os.path.abspath(video_path).startswith(os.path.abspath(TRAINING_VIDEOS_DIR)):
        return jsonify({'error': 'Training video not found'}), 404

    return send_file(video_path, mimetype='video/mp4')


@app.route('/admin/list-videos', methods=['GET'])
def admin_list_videos():
    """Admin endpoint to list videos in Railway volume"""
    admin_secret = os.getenv('ADMIN_SECRET', 'dev-secret-123')

    if request.args.get('secret') != admin_secret:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        static_videos = []
        training_videos = []

        if os.path.exists(STATIC_VIDEOS_DIR):
            static_videos = os.listdir(STATIC_VIDEOS_DIR)

        if os.path.exists(TRAINING_VIDEOS_DIR):
            training_videos = os.listdir(TRAINING_VIDEOS_DIR)

        return jsonify({
            'static_videos_dir': STATIC_VIDEOS_DIR,
            'static_videos': static_videos,
            'training_videos_dir': TRAINING_VIDEOS_DIR,
            'training_videos': training_videos,
            'is_railway': IS_RAILWAY,
            'data_dir': DATA_DIR if IS_RAILWAY else 'N/A'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/upload-video', methods=['POST'])
def admin_upload_video():
    """Admin endpoint to upload videos to Railway volume"""
    # Simple secret key auth - replace 'your-admin-secret' with a real secret
    admin_secret = os.getenv('ADMIN_SECRET', 'dev-secret-123')

    if request.headers.get('X-Admin-Secret') != admin_secret:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    video_type = request.form.get('type', 'static')  # 'static' or 'training'

    if video.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Sanitize filename
    filename = sanitize_filename(video.filename)

    # Only allow .mp4 files
    if not filename.endswith('.mp4'):
        return jsonify({'error': 'Only .mp4 files allowed'}), 400

    # Determine destination directory
    if video_type == 'training':
        dest_dir = TRAINING_VIDEOS_DIR
    else:
        dest_dir = STATIC_VIDEOS_DIR

    # Create directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(dest_dir, filename)
    video.save(file_path)

    return jsonify({
        'success': True,
        'filename': filename,
        'type': video_type,
        'path': file_path,
        'url': f'/training/{filename}' if video_type == 'training' else f'/{filename}'
    })


@app.route('/results/<filename>')
def serve_result(filename):
    """Serve processed result files and delete after sending"""
    print(f"[SERVE-RESULT] Request for file: {filename} from {request.remote_addr}")

    # Sanitize filename to prevent path traversal
    filename = sanitize_filename(filename)
    file_path = os.path.join(RESULT_DIR, filename)

    print(f"[SERVE-RESULT] Looking for file at: {file_path}")
    print(f"[SERVE-RESULT] RESULT_DIR: {RESULT_DIR}")
    print(f"[SERVE-RESULT] File exists: {os.path.exists(file_path)}")
    print(f"[SERVE-RESULT] RESULT_DIR contents: {os.listdir(RESULT_DIR) if os.path.exists(RESULT_DIR) else 'DIR NOT FOUND'}")

    # Verify file exists and is within result directory
    if not os.path.exists(file_path) or not os.path.abspath(file_path).startswith(os.path.abspath(RESULT_DIR)):
        print(f"[SERVE-RESULT] ❌ File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404

    # Send file with as_attachment to trigger download
    print(f"[SERVE-RESULT] ✅ Serving file: {filename}")
    response = send_file(file_path, as_attachment=True, download_name=f'cleaned_{filename}')

    # Schedule file deletion after response is sent
    @response.call_on_close
    def delete_files():
        try:
            # Delete the result file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[CLEANUP]  Deleted result: {filename}")

            # Delete corresponding upload file
            # Extract original task_id from processed filename
            original_name = filename.replace('_processed.avi', '.mp4')
            upload_path = os.path.join(UPLOAD_DIR, original_name)
            if os.path.exists(upload_path):
                os.remove(upload_path)
                print(f"[CLEANUP]  Deleted upload: {original_name}")
        except Exception as e:
            print(f"[WARNING]  Error deleting files: {e}")

    return response


@app.route('/demo_videos/<filename>')
def serve_demo_video(filename):
    """Serve demo/example videos from /data/demo_videos/ directory"""
    print(f"[DEMO-VIDEO] Request for: {filename}")

    # Sanitize filename
    filename = sanitize_filename(filename)
    demo_dir = os.path.join(DATA_DIR, 'demo_videos')
    file_path = os.path.join(demo_dir, filename)

    print(f"[DEMO-VIDEO] Serving from: {file_path}")

    if not os.path.exists(file_path):
        print(f"[DEMO-VIDEO] File not found: {file_path}")
        abort(404)

    # Verify file is within demo directory (security)
    if not os.path.abspath(file_path).startswith(os.path.abspath(demo_dir)):
        print(f"[DEMO-VIDEO] Path traversal attempt blocked")
        abort(403)

    return send_file(file_path, mimetype='video/mp4')


@app.route('/api/upload-result', methods=['POST', 'OPTIONS'])
def upload_result():
    if request.method == 'OPTIONS':
        return ('', 204)
    """Accept result file from a remote worker and store it under results/.

    Request: multipart/form-data with field 'file' and optional 'filename'.
    Response: { "status": "success", "result_url": "/results/<filename>" }
    """
    try:
        print(f"[UPLOAD-RESULT] Received upload request from {request.remote_addr}")

        if 'file' not in request.files:
            print(f"[UPLOAD-RESULT ERROR] No file in request")
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        up = request.files['file']
        if up.filename == '':
            print(f"[UPLOAD-RESULT ERROR] Empty filename")
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        req_filename = request.form.get('filename') or up.filename
        safe_name = sanitize_filename(req_filename)

        print(f"[UPLOAD-RESULT] Saving file: {safe_name} to {RESULT_DIR}")
        os.makedirs(RESULT_DIR, exist_ok=True)
        dest = os.path.join(RESULT_DIR, safe_name)

        up.save(dest)
        file_size_mb = os.path.getsize(dest) / (1024 * 1024)

        print(f"[UPLOAD-RESULT] ✅ Saved {safe_name} ({file_size_mb:.2f} MB) to {dest}")
        print(f"[UPLOAD-RESULT] File exists check: {os.path.exists(dest)}")
        print(f"[UPLOAD-RESULT] RESULT_DIR contents: {os.listdir(RESULT_DIR) if os.path.exists(RESULT_DIR) else 'DIR NOT FOUND'}")

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

        print(f"[OK] Received segment upload: video_id={video_id}, seg_idx={seg_idx}, file={safe_name}")

        return jsonify({
            'status': 'success',
            'segment_url': f'/results/{safe_name}',
            'video_id': video_id,
            'seg_idx': seg_idx
        })
    except Exception as e:
        print(f"[ERROR] Segment upload error: {e}")
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
        print(f"[ERROR] Error serving temp file {filepath}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/privacy')
def privacy_policy():
    """Serve Privacy Policy page"""
    return send_file(os.path.join(app.static_folder, 'privacy.html'))


@app.route('/terms')
def terms_of_service():
    """Serve Terms of Service page"""
    return send_file(os.path.join(app.static_folder, 'terms.html'))


@app.route('/premium')
@app.route('/premium.html')
def premium_page():
    """Serve Premium/Pricing page"""
    return send_file(os.path.join(app.static_folder, 'premium.html'))


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


@app.route('/api/sam2/select-object', methods=['POST'])
@require_auth
def sam2_select_object():
    """Interactive SAM2 object selection - uses local worker via Redis pub/sub"""
    try:
        data = request.json

        # Get frame data and points
        frame_base64 = data.get('frame_data')
        points = data.get('points', [])
        video_width = data.get('video_width')
        video_height = data.get('video_height')

        if not frame_base64 or not points:
            return jsonify({'status': 'error', 'message': 'Missing frame data or points'}), 400

        # Generate unique request ID
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        # Connect to Redis
        REDIS_URL = os.environ.get('REDIS_URL')
        if not REDIS_URL:
            return jsonify({'status': 'error', 'message': 'Redis not configured'}), 500

        print(f"[SAM2] Using Redis: {REDIS_URL[:50]}...")
        redis_client = redis.from_url(REDIS_URL, decode_responses=False)

        # Subscribe to response channel BEFORE publishing request
        response_channel = f'sam2:selection:response:{request_id}'
        pubsub = redis_client.pubsub()
        pubsub.subscribe(response_channel)

        # Publish request to local worker
        request_data = {
            'request_id': request_id,
            'frame_data': frame_base64,
            'points': points,
            'video_width': video_width,
            'video_height': video_height
        }

        print(f"[SAM2] Publishing request {request_id} to channel: sam2:selection:request")
        print(f"[SAM2] Request data: points={len(points)}, video={video_width}x{video_height}")
        redis_client.publish('sam2:selection:request', json.dumps(request_data))

        # Wait for response (timeout: 5 seconds)
        timeout = 5.0
        start_time = time.time()

        print(f"[SAM2] Waiting for response on: {response_channel}")
        while time.time() - start_time < timeout:
            message = pubsub.get_message(timeout=0.1)

            if message and message['type'] == 'message':
                response_data = json.loads(message['data'])

                pubsub.unsubscribe()
                pubsub.close()

                if response_data.get('status') == 'success':
                    return jsonify({
                        'status': 'success',
                        'mask': response_data['mask'],
                        'score': response_data.get('score')
                    })
                else:
                    return jsonify({
                        'status': 'error',
                        'message': response_data.get('error', 'Unknown error')
                    }), 500

        # Timeout
        pubsub.unsubscribe()
        pubsub.close()

        return jsonify({
            'status': 'error',
            'message': 'Local worker timeout - is SAM2 worker running?'
        }), 504

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500



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

# ===================================
# BILLING ENDPOINTS (Stripe)
# ===================================

# Stripe Price IDs for credit packages (set in environment)
STRIPE_PRICE_IDS = {
    'credits_10': os.getenv('STRIPE_PRICE_ID_CREDITS_10', ''),
    'credits_25': os.getenv('STRIPE_PRICE_ID_CREDITS_25', ''),
    'credits_100': os.getenv('STRIPE_PRICE_ID_CREDITS_100', ''),
    'credits_250': os.getenv('STRIPE_PRICE_ID_CREDITS_250', ''),
    'credits_500': os.getenv('STRIPE_PRICE_ID_CREDITS_500', ''),
    'starter': os.getenv('STRIPE_PRICE_ID_STARTER', ''),
    'pro': os.getenv('STRIPE_PRICE_ID_PRO', ''),
    'enterprise': os.getenv('STRIPE_PRICE_ID_ENTERPRISE', '')
}

# Credit amounts for each package
CREDIT_AMOUNTS = {
    'credits_10': 10,
    'credits_25': 25,
    'credits_100': 100,
    'credits_250': 250,
    'credits_500': 500,
    'starter': 20,
    'pro': 50,
    'enterprise': 300
}

@app.route('/api/billing/create-checkout-session', methods=['POST', 'OPTIONS'])
def create_checkout_session():
    """Create a Stripe checkout session for subscriptions or one-time credit purchases"""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not STRIPE_ENABLED:
        return jsonify({'error': 'Billing is not enabled on this server'}), 503

    try:
        data = request.get_json() or {}
        plan = data.get('plan')
        package = data.get('package')
        user_id = data.get('user_id')
        mode = data.get('mode', 'subscription')  # 'subscription' or 'payment'

        # Determine which price ID to use
        price_key = package if package else plan
        price_id = STRIPE_PRICE_IDS.get(price_key)

        if not price_id:
            return jsonify({'error': f'Invalid plan/package: {price_key}'}), 400

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Get base URL for success/cancel redirects
        base_url = request.host_url.rstrip('/')

        # Create checkout session
        checkout_params = {
            'payment_method_types': ['card'],
            'line_items': [{
                'price': price_id,
                'quantity': 1,
            }],
            'mode': mode,
            'success_url': f'{base_url}/success.html?session_id={{CHECKOUT_SESSION_ID}}',
            'cancel_url': f'{base_url}/premium.html',
            'client_reference_id': user_id,
            'metadata': {
                'user_id': user_id,
                'plan': plan if plan else '',
                'package': package if package else '',
            }
        }

        session = stripe.checkout.Session.create(**checkout_params)

        return jsonify({'url': session.url})

    except Exception as e:
        print(f"[BILLING-ERROR] {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/billing/webhook', methods=['POST'])
@app.route('/api/stripe/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for payment events"""
    if not STRIPE_ENABLED:
        return jsonify({'error': 'Billing is not enabled'}), 503

    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')

    try:
        # Verify webhook signature if secret is configured
        if webhook_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            event = json.loads(payload)
            print("[WARNING] Stripe webhook signature verification disabled (no STRIPE_WEBHOOK_SECRET)")

        event_type = event['type']
        data_object = event['data']['object']

        print(f"[STRIPE-WEBHOOK] Received: {event_type}")

        # Handle successful checkout (one-time or subscription)
        if event_type == 'checkout.session.completed':
            user_id = data_object.get('client_reference_id')
            metadata = data_object.get('metadata', {})
            package = metadata.get('package')
            plan = metadata.get('plan')

            # Determine credit amount
            key = package if package else plan
            credits_to_add = CREDIT_AMOUNTS.get(key, 0)

            if credits_to_add > 0 and user_id:
                # Award credits to user
                print(f"[BILLING] User {user_id} purchased {key}: +{credits_to_add} credits")
                if AUTH_ENABLED:
                    try:
                        with get_db() as conn:
                            cur = conn.cursor()
                            cur.execute(
                                'UPDATE users SET credits = credits + %s WHERE id = %s RETURNING credits',
                                (credits_to_add, user_id)
                            )
                            result = cur.fetchone()
                            if result:
                                new_balance = result[0]
                                print(f"[BILLING] ✅ Credits added successfully. User {user_id} new balance: {new_balance}")
                            else:
                                print(f"[BILLING] ❌ User {user_id} not found in database")
                    except Exception as e:
                        print(f"[BILLING] ❌ Failed to add credits for user {user_id}: {e}")
                        import traceback; traceback.print_exc()

        # Handle subscription renewals
        elif event_type == 'invoice.payment_succeeded':
            customer_id = data_object.get('customer')
            subscription_id = data_object.get('subscription')

            if subscription_id:
                # TODO: Award renewal credits based on subscription plan
                print(f"[BILLING] Subscription {subscription_id} renewed for customer {customer_id}")

        # Handle subscription updates/cancellations
        elif event_type in ['customer.subscription.updated', 'customer.subscription.deleted']:
            subscription = data_object
            customer_id = subscription.get('customer')
            status = subscription.get('status')

            print(f"[BILLING] Subscription for customer {customer_id} is now: {status}")

        return jsonify({'status': 'success'})

    except Exception as e:
        print(f"[WEBHOOK-ERROR] {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/api/billing/create-portal-session', methods=['POST', 'OPTIONS'])
def create_portal_session():
    """Create a Stripe billing portal session for subscription management"""
    if request.method == 'OPTIONS':
        return ('', 204)

    if not STRIPE_ENABLED:
        return jsonify({'error': 'Billing is not enabled'}), 503

    try:
        data = request.get_json() or {}
        customer_id = data.get('customer_id')

        if not customer_id:
            return jsonify({'error': 'customer_id is required'}), 400

        # Get base URL for return redirect
        base_url = request.host_url.rstrip('/')

        # Create portal session
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f'{base_url}/premium.html',
        )

        return jsonify({'url': session.url})

    except Exception as e:
        print(f"[BILLING-PORTAL-ERROR] {e}")
        return jsonify({'error': str(e)}), 500


# Explicit CORS preflight catch-all for /api/* (helps when proxies strip headers)
@app.route('/api/<path:subpath>', methods=['OPTIONS'])
def cors_preflight(subpath):
    return ('', 204)
