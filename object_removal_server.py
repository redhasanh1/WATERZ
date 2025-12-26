"""
Object Removal Server - Local Flask API for SAM2 tracking + export

Workflow:
1. User uploads video
2. User clicks to select object (SAM2 TensorRT)
3. Track object through full video
4. Export with background/format options
"""

import os
import sys
import uuid
import json
import time
import shutil
import threading
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "temp_uploads"
MASKS_DIR = BASE_DIR / "temp_masks"
OUTPUT_DIR = BASE_DIR / "temp_output"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for d in [UPLOAD_DIR, MASKS_DIR, OUTPUT_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# Job tracking
jobs = {}  # job_id -> {status, progress, message, result_path, ...}

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'wmv', 'flv', 'mpeg', 'mpg', '3gp', 'm4v', 'ts', 'ogv', 'mts', 'm2ts'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# Static file serving
# ============================================================
@app.route('/')
def serve_index():
    return send_from_directory('web', 'object-removal.html')

@app.route('/backgroundremover')
def serve_backgroundremover():
    """Secret route for background remover (production)"""
    return send_from_directory('web', 'object-removal.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)

@app.route('/temp_output/<path:path>')
def serve_temp_output(path):
    return send_from_directory(OUTPUT_DIR, path)


# ============================================================
# Upload endpoint
# ============================================================
@app.route('/api/object-removal/upload', methods=['POST'])
def upload_video():
    """Upload video for object removal"""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

    # Generate unique job ID
    job_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()

    # Save video
    video_path = UPLOAD_DIR / f"{job_id}.{ext}"
    file.save(str(video_path))

    # Get video info
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # Initialize job
    jobs[job_id] = {
        'status': 'uploaded',
        'video_path': str(video_path),
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'points': [],
        'masks_dir': None,
        'result_path': None
    }

    print(f"[UPLOAD] Job {job_id}: {width}x{height} @ {fps:.1f}fps, {frame_count} frames")

    return jsonify({
        'status': 'success',
        'job_id': job_id,
        'video_url': f'/api/object-removal/video/{job_id}',
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    })


@app.route('/api/object-removal/video/<job_id>')
def serve_video(job_id):
    """Serve uploaded video for preview"""
    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    video_path = jobs[job_id]['video_path']
    return send_file(video_path, mimetype='video/mp4')


# ============================================================
# SAM2 Selection (store points for tracking)
# ============================================================
@app.route('/api/object-removal/select', methods=['POST'])
def select_object():
    """Store clicked point for SAM2 tracking (mask generated during tracking via WSL2)"""
    data = request.json
    job_id = data.get('job_id')
    points = data.get('points', [])  # [{x, y, label}, ...]
    frame_index = data.get('frame_index', 0)

    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]

    # Store points for later tracking
    job['points'] = points
    job['frame_index'] = frame_index

    print(f"[SELECT] Job {job_id}: Points {points} at frame {frame_index}")

    # Return success - mask will be generated during tracking via WSL2
    # No instant preview since SAM2 TensorRT runs in WSL2/Docker, not Windows
    return jsonify({
        'status': 'success',
        'message': 'Point selected - click "Remove Selected" to track',
        'points': points,
        'frame_index': frame_index,
        'width': job['width'],
        'height': job['height']
    })


# ============================================================
# SAM2 TensorRT Interactive Selection (instant mask preview)
# ============================================================
@app.route('/api/sam2/select-object', methods=['POST'])
def sam2_select_object():
    """Interactive SAM2 object selection - uses TensorRT worker via Redis"""
    import redis

    try:
        data = request.json
        frame_base64 = data.get('frame_data')
        points = data.get('points', [])
        video_width = data.get('video_width')
        video_height = data.get('video_height')

        if not frame_base64 or not points:
            return jsonify({'status': 'error', 'message': 'Missing frame data or points'}), 400

        # Get Redis URL
        redis_url = 'redis://:watermarkz_secure_2024@localhost:6379/0'
        if os.path.exists('redis_url.txt'):
            with open('redis_url.txt', 'r') as f:
                redis_url = f.read().strip()

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        redis_client = redis.from_url(redis_url, decode_responses=False)

        # Subscribe to response channel BEFORE pushing request
        response_channel = f'sam2:selection:response:{request_id}'
        pubsub = redis_client.pubsub()
        pubsub.subscribe(response_channel)

        # Push request to list (worker uses BRPOP)
        request_data = {
            'request_id': request_id,
            'frame_data': frame_base64,
            'points': points,
            'video_width': video_width,
            'video_height': video_height
        }
        print(f"[SAM2] Pushing request {request_id} to sam2:selection:request")
        redis_client.lpush('sam2:selection:request', json.dumps(request_data))

        # Wait for response (5s timeout)
        timeout = 5.0
        start_time = time.time()

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

        pubsub.unsubscribe()
        pubsub.close()
        return jsonify({
            'status': 'error',
            'message': 'SAM2 worker timeout - is start_object_server.py running?'
        }), 504

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================
# Auto-Detect Objects (YOLO)
# ============================================================
@app.route('/api/object-removal/auto-detect', methods=['POST'])
def auto_detect():
    """Use YOLO to detect objects in first frame"""
    import cv2

    data = request.json
    job_id = data.get('job_id')

    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]
    video_path = job['video_path']

    try:
        # Read first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'status': 'error', 'message': 'Failed to read video frame'}), 500

        # Run YOLO detection
        from ultralytics import YOLO

        # Try Windows path first, then WSL path
        model_path = r"D:\watermarkz\runs\detect\sora_watermark_v2\weights\best.pt"
        if not os.path.exists(model_path):
            model_path = '/mnt/d/watermarkz/runs/detect/sora_watermark_v2/weights/best.pt'

        model = YOLO(model_path)
        results = model(frame, conf=0.3, device='cuda', verbose=False)

        detections = []
        height, width = frame.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append({
                'x': float(cx),
                'y': float(cy),
                'x_percent': float(cx / width * 100),
                'y_percent': float(cy / height * 100),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'conf': float(box.conf[0])
            })

        print(f"[YOLO] Detected {len(detections)} objects in job {job_id}")
        return jsonify({
            'status': 'success',
            'detections': detections,
            'width': width,
            'height': height
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================
# Track full video (via Celery queue)
# ============================================================
def get_celery():
    """Get Celery app instance"""
    from celery import Celery
    redis_url = 'redis://:watermarkz_secure_2024@localhost:6379/0'
    if os.path.exists('redis_url.txt'):
        with open('redis_url.txt', 'r') as f:
            redis_url = f.read().strip()
    return Celery('watermark', broker=redis_url, backend=redis_url)

@app.route('/api/object-removal/track', methods=['POST'])
def track_video():
    """Start full video tracking with SAM2 via Celery queue"""
    data = request.json
    job_id = data.get('job_id')
    modified_masks = data.get('modified_masks')  # User-drawn mask modifications

    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]

    if not job.get('points'):
        return jsonify({'status': 'error', 'message': 'No points selected'}), 400

    # Update status
    job['status'] = 'tracking'
    job['progress'] = 0
    job['message'] = 'Submitting to SAM2 worker...'

    # Use WSL-native temp path for masks (avoids read-only mount issues)
    # WSL2 writes to /tmp, Windows reads via \\wsl$\Ubuntu\tmp\...
    wsl_masks_dir = f"/tmp/sam2_masks/{job_id}"
    windows_masks_dir = f"\\\\wsl$\\Ubuntu\\tmp\\sam2_masks\\{job_id}"
    job['masks_dir'] = windows_masks_dir  # Windows UNC path for reading
    job['wsl_masks_dir'] = wsl_masks_dir  # WSL path for worker

    try:
        from celery import signature

        video_path = job['video_path']
        points = job['points']
        frame_idx = job.get('frame_index', 0)

        # Convert points to list of (x, y) tuples
        point_coords = [(p['x'], p['y']) for p in points]
        labels = [p.get('label', 1) for p in points]

        celery = get_celery()

        # Submit to wsl_sam2 queue - use WSL-native path for masks
        s1 = signature(
            'sam2.generate_masks_fullfps',
            args=[video_path, wsl_masks_dir],  # WSL path - always writable
            kwargs={
                'prompt_mode': 'point',
                'points': point_coords,
                'labels': labels,
                'frame_idx': frame_idx,
                'api_base': None,
                'modified_masks': modified_masks
            },
            queue='wsl_sam2_local'
        )

        # Generate a unique task ID
        task_id = f"objrem_{job_id}"
        result = s1.apply_async(task_id=task_id)

        job['celery_task_id'] = task_id
        job['message'] = 'Tracking in progress...'

        print(f"[TRACK] Submitted Celery task {task_id} for job {job_id}")
        print(f"[TRACK] Points: {point_coords}, Frame: {frame_idx}")

        return jsonify({
            'status': 'success',
            'message': 'Tracking started via Celery',
            'job_id': job_id,
            'task_id': task_id
        })

    except Exception as e:
        print(f"[TRACK] Error: {e}")
        import traceback
        traceback.print_exc()
        job['status'] = 'error'
        job['message'] = str(e)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================
# Check job status (polls Celery AsyncResult)
# ============================================================
@app.route('/api/object-removal/status/<job_id>')
def get_status(job_id):
    """Get job status and progress from Celery task"""
    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]
    total_frames = job.get('frame_count', 0)

    # If we have a Celery task, poll its status
    celery_task_id = job.get('celery_task_id')
    if celery_task_id and job.get('status') == 'tracking':
        completed_via_redis = False

        # First, check Redis hash for completion (WSL worker updates this directly)
        try:
            redis_url = 'redis://:watermarkz_secure_2024@localhost:6379/0'
            if os.path.exists('redis_url.txt'):
                with open('redis_url.txt', 'r') as f:
                    redis_url = f.read().strip()
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_data = redis_client.hgetall(f"objrem:{job_id}")
            if redis_data and redis_data.get('masks_dir'):
                # WSL worker completed! Update local job state
                job['status'] = 'completed'
                job['progress'] = 100
                job['message'] = 'Tracking complete!'
                job['masks_dir'] = redis_data.get('masks_dir')
                # DON'T overwrite video_path - Redis has CDN URL but we need local path for cv2
                completed_via_redis = True
                print(f"[STATUS] Task {celery_task_id} completed (detected via Redis hash)")
        except Exception as e:
            print(f"[STATUS] Redis check error: {e}")

        # Fallback: Check Celery AsyncResult
        if not completed_via_redis:
            try:
                from celery.result import AsyncResult
                celery = get_celery()
                result = AsyncResult(celery_task_id, app=celery)

                if result.ready():
                    # Task completed
                    if result.successful():
                        job['status'] = 'completed'
                        job['progress'] = 100
                        job['message'] = 'Tracking complete!'
                        print(f"[STATUS] Task {celery_task_id} completed successfully")
                    else:
                        job['status'] = 'error'
                        job['message'] = str(result.result) if result.result else 'Task failed'
                        print(f"[STATUS] Task {celery_task_id} failed: {result.result}")
                elif result.state == 'PROCESSING':
                    # Task in progress - get meta info
                    meta = result.info or {}
                    job['progress'] = meta.get('progress', 0)
                    job['message'] = meta.get('status', 'Processing...')
                elif result.state == 'PENDING':
                    job['message'] = 'Waiting for worker...'
                elif result.state == 'FAILURE':
                    job['status'] = 'error'
                    job['message'] = str(result.result) if result.result else 'Task failed'

            except Exception as e:
                print(f"[STATUS] Celery poll error: {e}")

    # Calculate current frame from progress
    progress = job.get('progress', 0)
    current_frame = int(total_frames * progress / 100) if total_frames > 0 else 0

    return jsonify({
        'status': job.get('status', 'unknown'),
        'progress': progress,
        'export_progress': job.get('export_progress', progress),
        'message': job.get('message', ''),
        'current_frame': current_frame,
        'total_frames': total_frames,
        'result_url': f'/api/object-removal/download/{job_id}' if job.get('result_path') else None,
        'cdn_url': job.get('cdn_url')  # B2 cloud URL (production)
    })


# ============================================================
# Export with options
# ============================================================
def apply_operation(frame, mask, operation, options):
    """Apply operation to frame using mask (from video_tools_gui.py)"""
    import cv2
    import numpy as np

    # Ensure mask matches frame size
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Apply dilation if set
    dilation = options.get('dilation', 0)
    if dilation > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilation)

    mask_f = mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_f] * 3, axis=-1)

    color_bgr = options.get('color_bgr', (0, 255, 0))
    blur_amount = options.get('blur_amount', 25)
    bg_type = options.get('bg_type', 'color')

    # Ensure blur kernel is odd
    blur_kernel = blur_amount * 2 + 1

    if operation == 'keep_object':
        # Keep inside mask, replace outside
        if bg_type == 'transparent':
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask
            return rgba
        elif bg_type == 'color':
            bg = np.full_like(frame, color_bgr, dtype=np.uint8)
            return (frame * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
        else:  # blur
            blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
            return (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'remove_object':
        # Keep outside mask, fill inside with color/blur/transparent
        inv_mask_3ch = 1 - mask_3ch
        if bg_type == 'transparent':
            # Object area becomes transparent, background stays visible
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255 - mask  # Invert: mask area = transparent
            return rgba
        elif bg_type == 'color':
            fill = np.full_like(frame, color_bgr, dtype=np.uint8)
        else:  # blur
            fill = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        return (frame * inv_mask_3ch + fill * mask_3ch).astype(np.uint8)

    elif operation == 'fill_inside':
        fill = np.full_like(frame, color_bgr, dtype=np.uint8)
        return (fill * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'fill_outside':
        if bg_type == 'transparent':
            # Outside becomes transparent, inside stays visible
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask  # Mask area = opaque, outside = transparent
            return rgba
        fill = np.full_like(frame, color_bgr, dtype=np.uint8)
        return (frame * mask_3ch + fill * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'blur_inside':
        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        return (blurred * mask_3ch + frame * (1 - mask_3ch)).astype(np.uint8)

    elif operation == 'blur_outside':
        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        return (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

    return frame


@app.route('/api/object-removal/export', methods=['POST'])
def export_video():
    """Export video with background/format options"""
    data = request.json
    job_id = data.get('job_id')

    # Export options
    operation = data.get('operation', 'keep_object')  # keep_object, remove_object, fill_inside, fill_outside, blur_inside, blur_outside
    background = data.get('background', 'transparent')  # transparent, color, blur
    bg_color = data.get('bg_color', '#00FF00')  # hex color
    output_format = data.get('format', 'mp4')  # mp4, webm, png, png_sequence, gif

    # Normalize format aliases
    if output_format == 'png_sequence':
        output_format = 'png'  # Same as png
    blur_amount = data.get('blur_amount', data.get('blur', 20))  # blur strength
    dilation = data.get('dilation', 0)  # mask dilation

    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]

    if job.get('status') != 'completed':
        return jsonify({'status': 'error', 'message': 'Video not tracked yet'}), 400

    # Update status
    job['status'] = 'exporting'
    job['progress'] = 0
    job['message'] = 'Exporting...'

    def run_export():
        try:
            import cv2
            import numpy as np

            video_path = job['video_path']
            masks_dir = Path(job['masks_dir'])

            # Read video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Parse background color
            hex_color = bg_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            color_bgr = (b, g, r)

            # Options for apply_operation
            options = {
                'bg_type': background,
                'color_bgr': color_bgr,
                'blur_amount': blur_amount,
                'dilation': dilation
            }

            # Determine if we need transparent output
            needs_alpha = (operation == 'keep_object' and background == 'transparent') or output_format in ('webm', 'png')

            if output_format == 'png':
                # PNG sequence export
                output_dir = RESULTS_DIR / f"{job_id}_png"
                output_dir.mkdir(exist_ok=True)

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    mask_path = masks_dir / f"{frame_idx:05d}.png"
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            result = apply_operation(frame, mask, operation, options)
                            cv2.imwrite(str(output_dir / f"{frame_idx:05d}.png"), result)
                        else:
                            # No valid mask - create RGBA with full opacity if transparent mode
                            if background == 'transparent':
                                rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                                rgba[:, :, 3] = 255
                                cv2.imwrite(str(output_dir / f"{frame_idx:05d}.png"), rgba)
                            else:
                                cv2.imwrite(str(output_dir / f"{frame_idx:05d}.png"), frame)
                    else:
                        # No mask file - create RGBA with full opacity if transparent mode
                        if background == 'transparent':
                            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                            rgba[:, :, 3] = 255
                            cv2.imwrite(str(output_dir / f"{frame_idx:05d}.png"), rgba)
                        else:
                            cv2.imwrite(str(output_dir / f"{frame_idx:05d}.png"), frame)

                    job['progress'] = int((frame_idx + 1) / frame_count * 100)

                cap.release()
                output_path = output_dir

            elif needs_alpha and output_format == 'webm':
                # WebM with alpha - use ffmpeg
                temp_frames_dir = OUTPUT_DIR / f"{job_id}_frames"
                temp_frames_dir.mkdir(exist_ok=True)
                output_path = RESULTS_DIR / f"{job_id}_output.webm"

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    mask_path = masks_dir / f"{frame_idx:05d}.png"
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            result = apply_operation(frame, mask, operation, options)
                            cv2.imwrite(str(temp_frames_dir / f"{frame_idx:05d}.png"), result)
                        else:
                            # No valid mask - create RGBA with full opacity
                            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                            rgba[:, :, 3] = 255
                            cv2.imwrite(str(temp_frames_dir / f"{frame_idx:05d}.png"), rgba)
                    else:
                        # No mask file - create RGBA with full opacity
                        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                        rgba[:, :, 3] = 255
                        cv2.imwrite(str(temp_frames_dir / f"{frame_idx:05d}.png"), rgba)

                    job['progress'] = int((frame_idx + 1) / frame_count * 50)

                cap.release()

                # Use ffmpeg to create WebM with alpha
                ffmpeg_cmd = f'ffmpeg -y -framerate {fps} -i "{temp_frames_dir}/%05d.png" -c:v libvpx-vp9 -pix_fmt yuva420p "{output_path}"'
                subprocess.run(ffmpeg_cmd, shell=True, check=True)

                # Cleanup temp frames
                shutil.rmtree(temp_frames_dir)

            else:
                # Regular MP4 export
                output_path = RESULTS_DIR / f"{job_id}_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    mask_path = masks_dir / f"{frame_idx:05d}.png"
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            result = apply_operation(frame, mask, operation, options)
                            # Drop alpha channel if present
                            if result.shape[2] == 4:
                                result = result[:, :, :3]
                            writer.write(result)
                        else:
                            writer.write(frame)
                    else:
                        writer.write(frame)

                    job['progress'] = int((frame_idx + 1) / frame_count * 100)

                cap.release()
                writer.release()

            job['status'] = 'export_complete'
            job['progress'] = 100
            job['export_progress'] = 100
            job['message'] = 'Export complete!'
            job['result_path'] = str(output_path)
            print(f"[EXPORT] Job {job_id} complete: {output_path}")

            # Upload to B2 + Cloudflare CDN (production)
            cdn_url = None
            try:
                from b2sdk.v2 import B2Api, InMemoryAccountInfo
                import time as _upload_time

                B2_KEY_ID = os.getenv('B2_KEY_ID')
                B2_APP_KEY = os.getenv('B2_APP_KEY')
                B2_BUCKET = os.getenv('B2_BUCKET', 'watermarkz')
                B2_CDN_URL = os.getenv('B2_CDN_URL', 'https://markz.humblewoslayer.workers.dev')

                if B2_KEY_ID and B2_APP_KEY and os.getenv('SKIP_B2_UPLOAD') != '1':
                    timestamp = int(_upload_time.time())
                    remote_path = f"results/{timestamp}_{os.path.basename(str(output_path))}"

                    job['message'] = 'Uploading to cloud...'
                    print(f"[B2] Uploading to {B2_BUCKET}/{remote_path}...")
                    _b2_start = _upload_time.time()
                    info = InMemoryAccountInfo()
                    b2_api = B2Api(info)
                    b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
                    bucket = b2_api.get_bucket_by_name(B2_BUCKET)

                    if output_format == 'png':
                        # ZIP the PNG folder first
                        import zipfile
                        zip_path = str(output_path) + '.zip'
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                            for png in Path(output_path).glob('*.png'):
                                zf.write(png, png.name)
                        bucket.upload_local_file(local_file=zip_path, file_name=remote_path + '.zip')
                        cdn_url = f"{B2_CDN_URL}/{remote_path}.zip"
                    else:
                        bucket.upload_local_file(local_file=str(output_path), file_name=remote_path)
                        cdn_url = f"{B2_CDN_URL}/{remote_path}"

                    _b2_time = _upload_time.time() - _b2_start
                    print(f"[B2] Upload complete in {_b2_time:.1f}s - CDN URL: {cdn_url}")
                    job['cdn_url'] = cdn_url
                    job['message'] = 'Export complete! (uploaded to cloud)'
                else:
                    print(f"[B2] Skipping upload (SKIP_B2_UPLOAD=1 or no credentials)")
            except ImportError:
                print(f"[B2] b2sdk not installed - skipping upload")
            except Exception as e:
                print(f"[B2] Upload failed: {e}")

        except Exception as e:
            print(f"[EXPORT] Error: {e}")
            import traceback
            traceback.print_exc()
            job['status'] = 'error'
            job['message'] = str(e)

    thread = threading.Thread(target=run_export, daemon=True)
    thread.start()

    return jsonify({
        'status': 'success',
        'message': 'Export started',
        'job_id': job_id
    })


# ============================================================
# Preview single frame
# ============================================================
@app.route('/api/object-removal/preview', methods=['POST'])
def preview_frame():
    """Generate preview of operation on single frame"""
    import cv2
    import numpy as np

    data = request.json
    job_id = data.get('job_id')

    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]

    if job.get('status') != 'completed':
        return jsonify({'status': 'error', 'message': 'Video not tracked yet'}), 400

    # Get parameters
    operation = data.get('operation', 'keep_object')
    background = data.get('background', 'transparent')
    bg_color = data.get('bg_color', '#00FF00')
    blur_amount = data.get('blur_amount', 20)
    dilation = data.get('dilation', 0)
    frame_index = data.get('frame_index', 0)

    try:
        video_path = job['video_path']
        masks_dir = Path(job['masks_dir'])

        # Parse color
        hex_color = bg_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)

        options = {
            'bg_type': background,
            'color_bgr': color_bgr,
            'blur_amount': blur_amount,
            'dilation': dilation
        }

        # Read frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'status': 'error', 'message': 'Could not read frame'}), 400

        # Load mask
        mask_path = masks_dir / f"{frame_index:05d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                result = apply_operation(frame, mask, operation, options)

                # Save preview
                preview_path = OUTPUT_DIR / f"{job_id}_preview.png"
                cv2.imwrite(str(preview_path), result)

                return jsonify({
                    'status': 'success',
                    'preview_url': f'/temp_output/{job_id}_preview.png'
                })

        return jsonify({'status': 'error', 'message': 'No mask for this frame'}), 400

    except Exception as e:
        print(f"[PREVIEW] Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================
# Download result
# ============================================================
@app.route('/api/object-removal/download/<job_id>')
def download_result(job_id):
    """Download processed video"""
    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]

    if not job.get('result_path') or not os.path.exists(job['result_path']):
        return jsonify({'status': 'error', 'message': 'Result not ready'}), 404

    return send_file(
        job['result_path'],
        as_attachment=True,
        download_name=f"object_removal_{job_id}.mp4"
    )


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  OBJECT REMOVAL SERVER")
    print("=" * 60)
    print(f"  Upload dir: {UPLOAD_DIR}")
    print(f"  Masks dir:  {MASKS_DIR}")
    print(f"  Results:    {RESULTS_DIR}")
    print("=" * 60)
    print()
    print("  Open http://localhost:5000 in your browser")
    print()

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
