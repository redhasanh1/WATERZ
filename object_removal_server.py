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
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================
# Static file serving
# ============================================================
@app.route('/')
def serve_index():
    return send_from_directory('web', 'object-removal.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('web', path)


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
# Track full video
# ============================================================
@app.route('/api/object-removal/track', methods=['POST'])
def track_video():
    """Start full video tracking with SAM2"""
    data = request.json
    job_id = data.get('job_id')

    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]

    if not job.get('points'):
        return jsonify({'status': 'error', 'message': 'No points selected'}), 400

    # Update status
    job['status'] = 'tracking'
    job['progress'] = 0
    job['message'] = 'Starting tracking...'

    # Create masks directory
    masks_dir = MASKS_DIR / f"tracked_{job_id}"
    masks_dir.mkdir(exist_ok=True)
    job['masks_dir'] = str(masks_dir)

    # Start tracking in background thread
    def run_tracking():
        try:
            video_path = job['video_path']
            points = job['points']
            frame_idx = job.get('frame_index', 0)

            # Convert to WSL path
            wsl_video = video_path.replace('D:\\', '/mnt/d/').replace('\\', '/')
            wsl_masks = str(masks_dir).replace('D:\\', '/mnt/d/').replace('\\', '/')

            # Build points argument
            points_str = ' '.join([f"--point {p['x']},{p['y']}" for p in points])

            # Run SAM2 tracking via WSL
            cmd = f'wsl -e bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && python sam2_track_wsl2.py \\"{wsl_video}\\" {points_str} \\"{wsl_masks}\\" --frame {frame_idx}"'

            print(f"[TRACK] Running: {cmd}")
            job['message'] = 'Tracking object...'

            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'  # Replace undecodable chars instead of crashing
            )

            # Monitor progress
            for line in process.stdout:
                print(f"[SAM2] {line.strip()}")
                # Parse progress from output
                if 'Forward' in line and '%' in line:
                    try:
                        pct = int(line.split('%')[0].split()[-1])
                        job['progress'] = pct
                    except:
                        pass

            process.wait()

            if process.returncode == 0:
                job['status'] = 'completed'  # Frontend expects 'completed'
                job['progress'] = 100
                job['message'] = 'Tracking complete!'
                print(f"[TRACK] Job {job_id} complete")
            else:
                job['status'] = 'error'
                job['message'] = 'Tracking failed'

        except Exception as e:
            print(f"[TRACK] Error: {e}")
            import traceback
            traceback.print_exc()
            job['status'] = 'error'
            job['message'] = str(e)

    thread = threading.Thread(target=run_tracking, daemon=True)
    thread.start()

    return jsonify({
        'status': 'success',
        'message': 'Tracking started',
        'job_id': job_id
    })


# ============================================================
# Check job status
# ============================================================
@app.route('/api/object-removal/status/<job_id>')
def get_status(job_id):
    """Get job status and progress"""
    if job_id not in jobs:
        return jsonify({'status': 'error', 'message': 'Job not found'}), 404

    job = jobs[job_id]

    # Calculate current frame from progress
    total_frames = job.get('frame_count', 0)
    progress = job.get('progress', 0)
    current_frame = int(total_frames * progress / 100) if total_frames > 0 else 0

    return jsonify({
        'status': job.get('status', 'unknown'),
        'progress': progress,
        'export_progress': job.get('export_progress', progress),
        'message': job.get('message', ''),
        'current_frame': current_frame,
        'total_frames': total_frames,
        'result_url': f'/api/object-removal/download/{job_id}' if job.get('result_path') else None
    })


# ============================================================
# Export with options
# ============================================================
@app.route('/api/object-removal/export', methods=['POST'])
def export_video():
    """Export video with background/format options"""
    data = request.json
    job_id = data.get('job_id')

    # Export options
    background = data.get('background', 'transparent')  # transparent, color, blur
    bg_color = data.get('bg_color', '#00FF00')  # hex color
    output_format = data.get('format', 'mp4')  # mp4, webm, png
    blur_amount = data.get('blur', 20)  # blur strength

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

            # Output path
            if output_format == 'webm':
                output_path = RESULTS_DIR / f"{job_id}_output.webm"
                fourcc = cv2.VideoWriter_fourcc(*'VP90')
            else:
                output_path = RESULTS_DIR / f"{job_id}_output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            # For transparent, we need to handle alpha differently
            if background == 'transparent' and output_format == 'webm':
                # WebM with alpha - use ffmpeg
                temp_frames_dir = OUTPUT_DIR / f"{job_id}_frames"
                temp_frames_dir.mkdir(exist_ok=True)

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Load mask
                    mask_path = masks_dir / f"{frame_idx:05d}.png"
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Resize mask if needed
                            if mask.shape[:2] != (height, width):
                                mask = cv2.resize(mask, (width, height))

                            # Create RGBA frame (object only with alpha)
                            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                            rgba[:, :, 3] = mask  # Alpha from mask

                            # Save as PNG with alpha
                            cv2.imwrite(str(temp_frames_dir / f"{frame_idx:05d}.png"), rgba)

                    job['progress'] = int((frame_idx + 1) / frame_count * 50)

                cap.release()

                # Use ffmpeg to create WebM with alpha
                ffmpeg_cmd = f'ffmpeg -y -framerate {fps} -i "{temp_frames_dir}/%05d.png" -c:v libvpx-vp9 -pix_fmt yuva420p "{output_path}"'
                subprocess.run(ffmpeg_cmd, shell=True, check=True)

                # Cleanup temp frames
                shutil.rmtree(temp_frames_dir)

            else:
                # Regular export (MP4 or colored background)
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                # Parse background color
                if background == 'color':
                    # Convert hex to BGR
                    hex_color = bg_color.lstrip('#')
                    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                    bg_bgr = np.array([b, g, r], dtype=np.uint8)

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Load mask
                    mask_path = masks_dir / f"{frame_idx:05d}.png"
                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Resize mask if needed
                            if mask.shape[:2] != (height, width):
                                mask = cv2.resize(mask, (width, height))

                            mask_float = mask.astype(np.float32) / 255.0
                            mask_3ch = np.stack([mask_float] * 3, axis=-1)

                            if background == 'color':
                                # Solid color background
                                bg = np.full_like(frame, bg_bgr)
                                output_frame = (frame * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
                            elif background == 'blur':
                                # Blurred background
                                blurred = cv2.GaussianBlur(frame, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
                                output_frame = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
                            else:
                                # Keep object only (black background)
                                output_frame = (frame * mask_3ch).astype(np.uint8)

                            writer.write(output_frame)
                        else:
                            writer.write(frame)
                    else:
                        writer.write(frame)

                    job['progress'] = int((frame_idx + 1) / frame_count * 100)

                cap.release()
                writer.release()

            job['status'] = 'export_complete'  # Frontend expects 'export_complete'
            job['progress'] = 100
            job['export_progress'] = 100
            job['message'] = 'Export complete!'
            job['result_path'] = str(output_path)
            print(f"[EXPORT] Job {job_id} complete: {output_path}")

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
