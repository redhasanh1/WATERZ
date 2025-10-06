"""
Production Server for Watermark Removal SaaS
- Async queue processing with Celery + Redis
- GPU-optimized YOLO + LaMa
- Keeps your PC usable while serving customers
- Designed for $1M/month scale
- ALL FILES STAY ON D DRIVE (inside watermarkz folder)
"""

import sys
import os

# CRITICAL: Force ALL temp/cache to D drive (watermarkz folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
UPLOAD_DIR = os.path.join(SCRIPT_DIR, 'uploads')
RESULT_DIR = os.path.join(SCRIPT_DIR, 'results')

# Create directories
for directory in [TEMP_DIR, CACHE_DIR, UPLOAD_DIR, RESULT_DIR]:
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

# Add python_packages to path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'python_packages'))

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

app = Flask(__name__, static_folder='web')
CORS(app)

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
    # Fix connection hanging
    broker_pool_limit=1,  # Limit connection pool
    broker_connection_timeout=3,  # 3 second timeout for broker connection
    result_backend_transport_options={'socket_connect_timeout': 3},
    task_ignore_result=False,  # We need results for status tracking
    task_acks_late=True,
)

# Global model instances (lazy loaded)
detector = None
inpainter = None

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

def get_models():
    """
    Lazy load models (called once per worker)
    Models are kept in GPU memory for fast inference
    """
    global detector, inpainter

    if detector is None or inpainter is None:
        print("=" * 60)
        print("Loading AI models...")
        print("=" * 60)

        # Use TensorRT YOLO for production speed
        from yolo_detector import YOLOWatermarkDetector
        detector = YOLOWatermarkDetector()
        print("✅ TensorRT YOLO loaded (GPU accelerated)")

        # Use optimized LaMa for 1.5-2x faster inpainting
        try:
            from lama_inpaint_optimized import LamaInpainterOptimized
            inpainter = LamaInpainterOptimized()
            print("✅ Optimized LaMa loaded (FP16 + CUDA, 1.5-2x faster)")
        except Exception as e:
            print(f"❌ Failed to load Optimized LaMa: {e}")
            print("⚠️  Falling back to standard LaMa...")
            try:
                from lama_inpaint_local import LamaInpainter
                inpainter = LamaInpainter()
            except Exception as e2:
                print(f"❌ Failed to load standard LaMa: {e2}")
                inpainter = None

        print("=" * 60)
        print("✅ Models loaded and ready!")
        print("=" * 60)

    return detector, inpainter


# ============================================================================
# Celery Tasks (Background Processing)
# ============================================================================

@celery.task(bind=True, name='watermark.remove_image')
def process_image_task(self, image_data, file_hash):
    """
    Background task for image watermark removal

    Args:
        image_data: Image bytes
        file_hash: MD5 hash for caching

    Returns:
        Processed image bytes
    """
    try:
        # Update progress
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Loading models'})

        # Load models (cached after first call)
        detector, inpainter = get_models()

        if inpainter is None:
            raise Exception("LaMa model not available")

        # Decode image
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Decoding image'})
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Failed to decode image")

        # Detect watermark (optimized settings)
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': 'Detecting watermark'})
        start_detect = time.time()
        detections = detector.detect(image, confidence_threshold=0.25, padding=0)
        detect_time = time.time() - start_detect

        # Remove watermark
        self.update_state(state='PROCESSING', meta={'progress': 60, 'status': 'Removing watermark'})

        if detections:
            start_inpaint = time.time()
            mask = detector.create_mask(image, detections)
            result = inpainter.inpaint_region(image, mask)
            inpaint_time = time.time() - start_inpaint
        else:
            result = image
            inpaint_time = 0
            print(f"⚠️  No watermark detected in image")

        # Encode result
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Encoding result'})
        _, buffer = cv2.imencode('.png', result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        result_bytes = buffer.tobytes()

        # Log performance
        total_time = detect_time + inpaint_time
        print(f"✅ Image processed in {total_time:.2f}s (detect: {detect_time:.2f}s, inpaint: {inpaint_time:.2f}s)")
        print(f"   Detections: {len(detections)}")
        print(f"   Size: {len(image_data)/1024:.1f}KB → {len(result_bytes)/1024:.1f}KB")

        return {
            'data': result_bytes,
            'metadata': {
                'detections': len(detections),
                'processing_time': total_time,
                'detect_time': detect_time,
                'inpaint_time': inpaint_time,
                'original_size': len(image_data),
                'result_size': len(result_bytes)
            }
        }

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        raise


@celery.task(bind=True, name='watermark.remove_video')
def process_video_task(self, video_path):
    """
    Background task for video watermark removal

    Args:
        video_path: Path to uploaded video

    Returns:
        Path to processed video
    """
    try:
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Loading models'})

        # Load models
        detector, inpainter = get_models()

        if inpainter is None:
            raise Exception("LaMa model not available")

        # Open video
        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Opening video'})

        # Check if file exists
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")

        print(f"Opening video: {video_path}")
        print(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}. OpenCV could not decode the file.")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup output (D drive only!)
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        output_filename = f'{name}_processed.avi'  # Use AVI with MJPG (most reliable)
        output_path = os.path.join(RESULT_DIR, output_filename)

        # Use MJPG codec (native to OpenCV, most reliable on Windows)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            raise Exception("Failed to create video writer")

        # Process frames
        frames_processed = 0
        frames_with_watermark = 0
        last_valid_bbox = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            progress = int((frames_processed / total_frames) * 85) + 10
            self.update_state(state='PROCESSING', meta={
                'progress': progress,
                'status': f'Processing frame {frames_processed}/{total_frames}'
            })

            # Detect watermark (optimized settings from test_video_removal.py)
            detections = detector.detect(frame, confidence_threshold=0.25, padding=0)

            if detections:
                frames_with_watermark += 1

                # Remove watermark
                mask = detector.create_mask(frame, detections)
                result = inpainter.inpaint_region(frame, mask)
                out.write(result)
            else:
                # No watermark, write original
                out.write(frame)

            frames_processed += 1

        # Cleanup
        cap.release()
        out.release()

        print(f"✅ Video processed: {frames_processed} frames, {frames_with_watermark} with watermarks")

        # Merge audio from original video using FFmpeg
        self.update_state(state='PROCESSING', meta={'progress': 95, 'status': 'Adding audio'})

        final_output = output_path.replace('.avi', '_with_audio.mp4')

        try:
            import subprocess

            # Use FFmpeg to copy audio from original and merge with processed video
            # -i input1: processed video (no audio)
            # -i input2: original video (with audio)
            # -map 0:v: take video from first input (processed)
            # -map 1:a: take audio from second input (original)
            # -c:v libx264: encode video with h264
            # -c:a aac: encode audio with aac
            # -shortest: match shortest stream duration

            # First check if original video has audio
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
                print(f"✅ Original video has audio - merging...")
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file
                    '-i', output_path,  # Processed video (no audio)
                    '-i', video_path,   # Original video (with audio)
                    '-map', '0:v:0',    # Video from processed (explicit stream)
                    '-map', '1:a:0',    # Audio from original (explicit stream)
                    '-c:v', 'libx264',  # Video codec
                    '-preset', 'ultrafast',  # Faster encoding
                    '-crf', '18',       # Better quality
                    '-c:a', 'aac',      # Audio codec
                    '-b:a', '192k',     # Audio bitrate
                    '-strict', 'experimental',  # Allow experimental AAC encoder
                    final_output
                ]
            else:
                print(f"⚠️  Original video has no audio - encoding without audio...")
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', output_path,
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '18',
                    final_output
                ]

            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"✅ Audio merged successfully")

                # Verify output actually has audio
                verify_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'a:0',
                    '-show_entries', 'stream=codec_type',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    final_output
                ]
                verify = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)

                if 'audio' in verify.stdout:
                    print(f"✅ VERIFIED: Output has audio track")
                    # Delete the temporary AVI file without audio
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    output_path = final_output
                else:
                    print(f"⚠️  WARNING: FFmpeg succeeded but output has NO audio!")
                    print(f"   This shouldn't happen. Check FFmpeg installation.")
                    print(f"   Stdout: {result.stdout}")
                    print(f"   Stderr: {result.stderr}")
            else:
                print(f"⚠️  FFmpeg audio merge FAILED!")
                print(f"   Return code: {result.returncode}")
                print(f"   Command: {' '.join(cmd)}")
                print(f"   Stderr: {result.stderr}")
                print(f"   Stdout: {result.stdout}")
                print(f"   Returning video without audio: {output_path}")
        except FileNotFoundError as e:
            print(f"⚠️  FFmpeg not found - returning video without audio")
            print(f"   Error: {e}")
            print(f"   Install FFmpeg: https://ffmpeg.org/download.html")
        except subprocess.TimeoutExpired:
            print(f"⚠️  FFmpeg timeout - audio merge took too long")
            print(f"   Returning video without audio")
        except Exception as e:
            print(f"⚠️  Error merging audio: {e}")
            print(f"   Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"   Returning video without audio")

        return {
            'path': output_path,
            'metadata': {
                'frames_processed': frames_processed,
                'frames_with_watermark': frames_with_watermark,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'has_audio': os.path.exists(final_output)
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
        else:
            # Queue image processing
            task = process_image_task.apply_async(args=[image_data, file_hash])

        return jsonify({
            'task_id': task.id,
            'status': 'queued',
            'file_type': 'video' if is_video else 'image'
        })

    except Exception as e:
        print(f"❌ Error queuing task: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """
    Check task status

    Returns:
        {
            'state': 'PENDING' | 'STARTED' | 'PROCESSING' | 'SUCCESS' | 'FAILURE',
            'progress': str,
            'result': { 'result_url': str } (if SUCCESS)
        }
    """
    from celery.result import AsyncResult

    task = AsyncResult(task_id, app=celery)

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
        # task.result is a dict with 'path' and 'metadata'
        result_data = task.result
        if isinstance(result_data, dict):
            result_path = result_data.get('path', result_data)
        else:
            result_path = result_data
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


@app.route('/api/download-from-url', methods=['POST'])
def download_from_url():
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


@app.route('/api/download-sora', methods=['POST'])
def download_sora():
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

@app.route('/api/upload', methods=['POST'])
def upload_file():
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


@app.route('/api/process', methods=['POST'])
def process_video():
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

        # Find video file with any extension
        video_path = None
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            test_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
            if os.path.exists(test_path):
                video_path = test_path
                break

        if not video_path:
            return jsonify({'status': 'error', 'message': 'Video not found'}), 404

        # Queue processing task
        print(f"📤 Queuing video processing task for: {video_path}")

        # Try direct Redis queue instead of apply_async
        print("🔄 Attempting to queue task via direct Redis push...")

        try:
            # Use send_task instead of apply_async - it's non-blocking
            import json as json_lib
            import kombu
            from kombu import Connection

            # Create direct connection to Redis
            with Connection(REDIS_URL, connect_timeout=2) as conn:
                # Generate task ID
                task_id = str(uuid.uuid4())

                # Create Celery-compatible message
                message = {
                    'id': task_id,
                    'task': 'watermark.remove_video',  # Must match @celery.task(name=...)
                    'args': [video_path],
                    'kwargs': {},
                    'retries': 0,
                    'eta': None,
                    'expires': None,
                }

                # Send to queue
                producer = conn.Producer(serializer='json')
                producer.publish(
                    json_lib.dumps(message),
                    exchange='',
                    routing_key='celery',
                    content_type='application/json',
                    content_encoding='utf-8',
                )

                print(f"✅ Task queued with ID: {task_id}")
                return jsonify({
                    'status': 'success',
                    'task_id': task_id
                })

        except Exception as e:
            print(f"❌ Failed to queue task: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': f'Failed to connect to Redis: {str(e)}'
            }), 500

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
