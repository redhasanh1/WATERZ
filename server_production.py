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

app = Flask(__name__, static_folder='web')
CORS(app)

# ============================================================================
# Configuration - ALL ON D DRIVE
# ============================================================================

# Redis configuration (for queue + caching)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

app.config['broker_url'] = REDIS_URL
app.config['result_backend'] = REDIS_URL
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR  # D drive only!
app.config['TEMP_FOLDER'] = TEMP_DIR  # D drive only!

# Initialize Celery
celery = Celery(app.name, broker=app.config['broker_url'])
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
)

# Global model instances (lazy loaded)
detector = None
inpainter = None

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

        try:
            # Try to use optimized versions
            from yolo_detector_optimized import OptimizedYOLODetector
            detector = OptimizedYOLODetector(
                model_path='yolov8n.pt',
                use_tensorrt=False  # Set to True after exporting to TensorRT
            )
        except ImportError:
            # Fallback to standard version
            from yolo_detector import YOLOWatermarkDetector
            detector = YOLOWatermarkDetector()
            print("⚠️  Using standard YOLO (slower)")

        try:
            from lama_inpaint_local import LamaInpainter
            inpainter = LamaInpainter()
        except Exception as e:
            print(f"❌ Failed to load LaMa: {e}")
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

        # Detect watermark
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': 'Detecting watermark'})
        start_detect = time.time()
        detections = detector.detect(image, confidence_threshold=0.3, padding=30)
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
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception("Failed to open video")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup output (D drive only!)
        filename = os.path.basename(video_path)
        output_filename = filename.replace('.', '_processed.')
        output_path = os.path.join(RESULT_DIR, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

            # Detect watermark
            detections = detector.detect(frame, confidence_threshold=0.3, padding=30)

            # Use temporal consistency (last known position)
            if not detections and last_valid_bbox:
                detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]

            if detections:
                frames_with_watermark += 1

                # Update last known position
                if detections[0]['confidence'] > 0.3:
                    last_valid_bbox = detections[0]['bbox']

                # Remove watermark
                try:
                    mask = detector.create_mask(frame, detections)
                    processed_frame = inpainter.inpaint_region(frame, mask)
                    out.write(processed_frame)
                except Exception as e:
                    print(f"⚠️  Frame {frames_processed} inpainting failed: {e}")
                    out.write(frame)
            else:
                out.write(frame)

            frames_processed += 1

        # Cleanup
        cap.release()
        out.release()

        print(f"✅ Video processed: {frames_processed} frames, {frames_with_watermark} with watermarks")

        return {
            'path': output_path,
            'metadata': {
                'frames_processed': frames_processed,
                'frames_with_watermark': frames_with_watermark,
                'fps': fps,
                'resolution': f"{width}x{height}"
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

@app.route('/')
def index():
    """Serve landing page"""
    return send_file('web/index.html')


@app.route('/web/<path:path>')
def serve_web(path):
    """Serve static files"""
    return send_file(f'web/{path}')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'models_loaded': detector is not None and inpainter is not None
    })


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
            'status': 'pending' | 'processing' | 'complete' | 'failed',
            'progress': 0-100,
            'message': str
        }
    """
    from celery.result import AsyncResult

    task = AsyncResult(task_id, app=celery)

    if task.state == 'PENDING':
        response = {
            'status': 'pending',
            'progress': 0,
            'message': 'Task is waiting in queue'
        }
    elif task.state == 'STARTED':
        info = task.info or {}
        response = {
            'status': 'processing',
            'progress': info.get('progress', 0),
            'message': info.get('status', 'Processing')
        }
    elif task.state == 'PROCESSING':
        response = {
            'status': 'processing',
            'progress': task.info.get('progress', 0),
            'message': task.info.get('status', 'Processing')
        }
    elif task.state == 'SUCCESS':
        response = {
            'status': 'complete',
            'progress': 100,
            'message': 'Processing complete'
        }
    else:
        # FAILURE or other error state
        response = {
            'status': 'failed',
            'progress': 0,
            'message': str(task.info)
        }

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
    Download video from URL (YouTube, TikTok, etc.)
    Uses yt-dlp to download the video

    Request: { "url": "https://youtube.com/..." }
    Response: { "status": "success", "task_id": "...", "video_url": "/uploads/..." }
    """
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

        # Generate unique filename
        task_id = str(uuid.uuid4())
        output_path = os.path.join(UPLOAD_DIR, f'{task_id}.mp4')

        # Download using yt-dlp Python module
        import yt_dlp

        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': False,  # Show errors
            'no_warnings': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'video_url': f'/uploads/{task_id}.mp4'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/download-sora', methods=['POST'])
def download_sora():
    """
    Download Sora video from OpenAI using Playwright bypass + cookies
    Bypasses Cloudflare protection using saved cookies

    Request: { "url": "https://openai.com/sora/..." }
    Response: { "status": "success", "task_id": "...", "video_url": "/uploads/..." }
    """
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'status': 'error', 'message': 'No URL provided'}), 400

        # Generate unique filename
        task_id = str(uuid.uuid4())
        output_path = os.path.join(UPLOAD_DIR, f'{task_id}.mp4')

        # Import Playwright
        from playwright.sync_api import sync_playwright
        import time
        import json

        # Path to cookies file
        cookies_file = os.path.join(SCRIPT_DIR, 'downz', 'cookies.json')

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
                browser.close()
                return jsonify({
                    'status': 'error',
                    'message': 'No cookies found. Run save_cookies.py first to authenticate.'
                }), 400

            page = context.new_page()

            # Hide webdriver
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            print(f"🌐 Navigating to: {url}")
            page.goto(url, wait_until='networkidle', timeout=60000)

            print("⏳ Waiting for page to load...")
            time.sleep(3)

            # Check if Cloudflare challenge appears
            if "cloudflare" in page.content().lower() or "just a moment" in page.content().lower():
                print("🔄 Cloudflare detected - waiting for it to resolve...")
                time.sleep(5)

            print("🔍 Looking for video element...")

            # Find video tag
            video_element = page.query_selector('video')

            if video_element:
                video_src = video_element.get_attribute('src')
                if not video_src:
                    source = page.query_selector('video source')
                    if source:
                        video_src = source.get_attribute('src')

                if video_src:
                    print(f"✅ Found video source: {video_src}")

                    # Make absolute URL if relative
                    if video_src.startswith('//'):
                        video_src = 'https:' + video_src
                    elif video_src.startswith('/'):
                        from urllib.parse import urljoin
                        video_src = urljoin(url, video_src)

                    print(f"⬇️  Downloading video...")

                    # Download using Playwright's request context (preserves cookies/auth)
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
                        browser.close()
                        return jsonify({
                            'status': 'error',
                            'message': f'Download failed: HTTP {response.status}'
                        }), 500
                else:
                    browser.close()
                    return jsonify({
                        'status': 'error',
                        'message': 'Could not find video source URL'
                    }), 404
            else:
                browser.close()
                return jsonify({
                    'status': 'error',
                    'message': 'No video element found on page'
                }), 404

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


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
    print("Starting Flask server on http://0.0.0.0:5000")
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
        host='127.0.0.1',
        port=9000,
        debug=False,  # Set to False for production
        threaded=True
    )
