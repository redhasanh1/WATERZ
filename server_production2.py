"""
SAM2 Watermark Removal Server - Parallel Production System
Isolated from YOLO production (server_production.py)

This server handles SAM2-based watermark removal using pre-generated masks.
"""

import os
import sys
import glob
import json
from flask import Flask, request, jsonify
from celery import Celery
import subprocess

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
PROPAINTER_DIR = os.path.join(BASE_DIR, 'faster-propainter-main')

# Add ProPainter to path
sys.path.insert(0, PROPAINTER_DIR)

# Create directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Load Redis URL
REDIS_URL = os.getenv('REDIS_URL', 'redis://:watermarkz_secure_2024@localhost:6379/0')
if os.path.exists('redis_url.txt'):
    with open('redis_url.txt', 'r') as f:
        REDIS_URL = f.read().strip()

# Configure Celery (new format)
celery = Celery(app.name, broker=REDIS_URL, backend=REDIS_URL)
celery.conf.task_track_started = True
try:
    # Allow tuning via environment; CLI flags still take precedence when provided.
    _pool = os.getenv('CELERY_POOL', None)
    _conc = os.getenv('CELERY_CONCURRENCY', os.getenv('WORKER_CONCURRENCY', None))
    if _pool:
        celery.conf.worker_pool = _pool
    if _conc:
        celery.conf.worker_concurrency = int(_conc)
except Exception:
    pass

# Get FFmpeg/FFprobe executables
def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths - check multiple locations"""
    # Check local ffmpeg folder first
    ffmpeg_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffmpeg.exe')
    ffprobe_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffprobe.exe')

    # Check static_ffmpeg package location
    if not os.path.exists(ffmpeg_exe):
        try:
            import static_ffmpeg
            static_path = os.path.dirname(static_ffmpeg.__file__)
            static_ffmpeg_exe = os.path.join(static_path, 'bin', 'win32', 'ffmpeg.exe')
            static_ffprobe_exe = os.path.join(static_path, 'bin', 'win32', 'ffprobe.exe')
            if os.path.exists(static_ffmpeg_exe):
                ffmpeg_exe = static_ffmpeg_exe
                ffprobe_exe = static_ffprobe_exe
                print(f"[FFmpeg] Using static_ffmpeg: {ffmpeg_exe}")
        except ImportError:
            pass

    # Final fallback to system PATH
    if not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = 'ffmpeg'
    if not os.path.exists(ffprobe_exe):
        ffprobe_exe = 'ffprobe'

    return ffmpeg_exe, ffprobe_exe

FFMPEG_EXE, FFPROBE_EXE = get_ffmpeg_executables()


def convert_to_10fps_gpu(video_path: str, output_path: str = None) -> str:
    """
    Convert video to 10fps using FFmpeg NVENC GPU acceleration.
    Returns path to the 10fps video.

    Args:
        video_path: Path to input video
        output_path: Optional output path (auto-generated if None)

    Returns:
        Path to 10fps video
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_10fps{ext}"

    # Try GPU acceleration first, fall back to CPU
    # Using scale filter to maintain quality while reducing framerate
    cmd = [
        str(FFMPEG_EXE),
        '-y',
        '-i', video_path,
        '-vf', 'fps=10',
        '-c:v', 'libx264',  # CPU fallback (more compatible)
        '-preset', 'ultrafast',
        '-crf', '18',
        '-an',  # No audio for processing
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"[ERROR] FFmpeg 10fps conversion failed: {result.stderr[:500]}")
        raise RuntimeError(f"10fps conversion failed: {result.stderr[:200]}")

    print(f"[OK] Converted to 10fps: {output_path}")
    return output_path


# Check ProPainter assets
def _check_propainter_assets():
    """Check if ProPainter model checkpoints exist"""
    required_files = [
        os.path.join(PROPAINTER_DIR, 'weights', 'ProPainter.pth'),
        os.path.join(PROPAINTER_DIR, 'weights', 'recurrent_flow_completion.pth')
    ]
    return all(os.path.exists(f) for f in required_files)

# Get ProPainter pipeline (cached at worker startup)
_propainter_pipeline = None

def get_propainter_pipeline():
    """Get cached ProPainter pipeline"""
    global _propainter_pipeline
    if _propainter_pipeline is None:
        from watermark import pipeline as faster_propainter_pipeline
        _propainter_pipeline = faster_propainter_pipeline
    return _propainter_pipeline

def clear_gpu_memory():
    """Clear GPU memory between segments"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

#============================================================================
# CELERY TASK: SAM2 Interactive Mode - 10fps OPTIMIZED Pipeline
#
# Optimizations:
# - Convert video to 10fps for SAM2 (3-6x faster mask generation)
# - Expand masks in memory using zero-copy references
# - Run ProPainter with in-memory arrays (ZERO disk I/O)
# - Use NeuFlow TRT for optical flow (10-70x faster than RAFT)
#============================================================================

@celery.task(bind=True, name='watermark.process_sam2_interactive')
def process_sam2_interactive_task(self, video_path, video_id=None, points=None, video_width=None, video_height=None, frame_index=0, api_base=None):
    """
    SAM2 Interactive Mode - 10fps OPTIMIZED Pipeline:
    - Downloads video from remote
    - Converts to 10fps for SAM2 (3-6x faster mask generation)
    - Generates masks at 10fps using SAM2Tracker
    - Expands masks to full FPS in memory (zero-copy)
    - Runs ProPainter with in-memory arrays (zero disk I/O)
    - Returns processed video with audio

    Args:
        video_path (str): Path to input video (on remote server)
        video_id (str): Video ID for tracking
        points (list): User selection points
        video_width (int): Width of the video
        video_height (int): Height of the video
        frame_index (int): The frame index the user clicked on (at original FPS)
    """
    try:
        import shutil
        import json
        import numpy as np
        import cv2
        import requests
        from urllib.parse import urljoin
        from watermark import expand_masks_10fps

        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Initializing SAM2 10fps Pipeline'})

        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing")

        # Generate video ID if not provided
        if not video_id:
            video_id = os.path.basename(video_path).split('.')[0][:8]

        # --- 1. Locate video (robust local/remote handling) ---
        from pathlib import PureWindowsPath

        # Extract filename from video_path (handles Windows paths)
        base_name = PureWindowsPath(video_path).name if '\\' in video_path else os.path.basename(video_path)
        UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
        local_video_path = os.path.join(UPLOAD_DIR, base_name)

        # Smart check: If file exists locally (same PC workers share uploads/), skip download!
        if os.path.exists(local_video_path):
            print(f"[SAM2] Video already exists locally: {local_video_path} (skip download)")
            video_path = local_video_path
        elif not os.path.exists(video_path):
            # File not found at provided path - try to download
            # 1) If provided path is an absolute URL, use it directly
            if isinstance(video_path, str) and video_path.lower().startswith(('http://', 'https://')):
                download_url = video_path
            else:
                # 2) Construct from API base or env TUNNEL_URL or optional file 'tunnel_url.txt'
                tunnel = api_base or os.getenv('TUNNEL_URL')
                if not tunnel:
                    for candidate in ('tunnel_url.txt', 'TUNNEL_URL.txt'):
                        fp = os.path.join(BASE_DIR, candidate)
                        if os.path.exists(fp):
                            try:
                                with open(fp, 'r') as fh:
                                    tunnel = fh.read().strip()
                                    if tunnel:
                                        print(f"[SAM2] Loaded TUNNEL_URL from {candidate}: {tunnel}")
                                        break
                            except Exception:
                                pass
                if not tunnel:
                    # 3) As a last resort, attempt to map '/data/uploads/<file>' → local uploads/<file>
                    #    If still missing, fail with actionable error
                    if os.path.exists(local_video_path):
                        video_path = local_video_path
                    else:
                        raise RuntimeError(f"Video not found locally and no TUNNEL_URL provided: {video_path}")
                else:
                    download_url = urljoin(tunnel.rstrip('/') + '/', f'uploads/{base_name}')

            if 'download_url' in locals():
                print(f"[SAM2] Downloading video: {download_url}")
                self.update_state(state='STARTED', meta={'progress': 1, 'status': 'Downloading video'})
                r = requests.get(download_url, headers={'ngrok-skip-browser-warning': 'true'}, timeout=120)
                r.raise_for_status()
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                with open(local_video_path, 'wb') as f:
                    f.write(r.content)
                print(f"[SAM2] Video downloaded to: {local_video_path}")
                video_path = local_video_path
        # else: video_path exists, use it as-is (local file path provided directly)

        # --- 2. Get video metadata (FPS, frame count) ---
        self.update_state(state='PROCESSING', meta={'progress': 3, 'status': 'Analyzing video'})
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_original = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"[SAM2] Video: {width}x{height} @ {original_fps}fps, {total_frames_original} frames")

        # --- 3. Convert video to 10fps for SAM2 (3-6x faster mask generation) ---
        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Converting to 10fps'})
        video_10fps_path = os.path.join(TEMP_DIR, f"{video_id}_10fps.mp4")

        if not os.path.exists(video_10fps_path):
            print(f"[SAM2] Converting to 10fps for faster mask generation...")
            video_10fps_path = convert_to_10fps_gpu(video_path, video_10fps_path)
        else:
            print(f"[SAM2] Using cached 10fps video: {video_10fps_path}")

        # --- 4. Extract 10fps frames for SAM2 ---
        self.update_state(state='PROCESSING', meta={'progress': 8, 'status': 'Extracting 10fps frames'})
        print(f"[SAM2] Extracting 10fps frames...")
        cap_10fps = cv2.VideoCapture(video_10fps_path)
        frames_10fps = []
        while True:
            ret, frame = cap_10fps.read()
            if not ret:
                break
            frames_10fps.append(frame)
        cap_10fps.release()

        print(f"[SAM2] Extracted {len(frames_10fps)} frames at 10fps (vs {total_frames_original} at original)")

        # --- 5. Generate masks at 10fps using WSL2 SAM2 subprocess ---
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Generating masks with SAM2 (10fps via WSL2)'})

        if not points:
            raise ValueError("No points provided for mask generation.")

        # Convert points to a bounding box for the tracker
        # Accepts the following formats:
        # - list of [x, y]
        # - list of [x, y, ...] (extra values ignored)
        # - list of {x: ..., y: ...}
        # - flat list [x1, y1, x2, y2, ...] (pairs)
        # - dict with key 'bbox' as [x1, y1, x2, y2]

        def _extract_xy_array(pts):
            # dict with bbox provided directly
            if isinstance(pts, dict):
                if 'bbox' in pts and isinstance(pts['bbox'], (list, tuple)) and len(pts['bbox']) >= 4:
                    bx1, by1, bx2, by2 = pts['bbox'][:4]
                    return np.array([[bx1, by1], [bx2, by2]], dtype=float)
                if 'points' in pts:
                    pts = pts['points']
                else:
                    raise ValueError('Invalid points format: dict missing bbox/points')

            # list/tuple handling
            if isinstance(pts, (list, tuple)) and len(pts) > 0:
                first = pts[0]
                # list of dicts
                if isinstance(first, dict):
                    xy = []
                    for p in pts:
                        xv = p.get('x', p.get('X', p.get('cx')))
                        yv = p.get('y', p.get('Y', p.get('cy')))
                        if xv is None or yv is None:
                            raise ValueError('Invalid points format: dict items must contain x/y')
                        xy.append([xv, yv])
                    return np.array(xy, dtype=float)
                # list of lists/tuples/arrays
                if isinstance(first, (list, tuple)):
                    arr = np.array(pts, dtype=float)
                    # ignore extras beyond first two columns
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        return arr[:, :2]
                    # if nested but not 2D as expected
                    raise ValueError('Invalid points format: nested lists must be of shape (N, >=2)')
                # flat list of numbers
                if isinstance(first, (int, float)):
                    arr = np.array(pts, dtype=float)
                    if arr.size == 2:
                        return arr.reshape(1, 2)
                    if arr.size >= 4:
                        xs = arr[0::2]
                        ys = arr[1::2]
                        return np.stack([xs, ys], axis=1)
            raise ValueError('Invalid points format')

        xy = _extract_xy_array(points)

        # If points look normalized [0..1], scale to pixel space
        try:
            if xy.size >= 2:
                max_x = float(np.max(xy[:, 0]))
                max_y = float(np.max(xy[:, 1]))
                min_x = float(np.min(xy[:, 0]))
                min_y = float(np.min(xy[:, 1]))
                # Heuristic: values within [0,1] likely normalized
                if 0.0 <= min_x <= 1.0 and 0.0 <= max_x <= 1.0 and 0.0 <= min_y <= 1.0 and 0.0 <= max_y <= 1.0:
                    px_w = video_width or width
                    px_h = video_height or height
                    xy[:, 0] = xy[:, 0] * px_w
                    xy[:, 1] = xy[:, 1] * px_h
        except Exception:
            # Non-fatal; fallback to raw values
            pass

        x1, y1 = np.min(xy, axis=0)
        x2, y2 = np.max(xy, axis=0)

        # Ensure bbox has non-zero area; expand a bit if needed
        if int(x1) == int(x2) and int(y1) == int(y2):
            pad = 8
            x1 -= pad
            y1 -= pad
            x2 += pad
            y2 += pad

        # Clip to video bounds
        px_w = video_width or width
        px_h = video_height or height
        x1 = max(0, min(int(round(x1)), int(px_w) - 1))
        y1 = max(0, min(int(round(y1)), int(px_h) - 1))
        x2 = max(0, min(int(round(x2)), int(px_w) - 1))
        y2 = max(0, min(int(round(y2)), int(px_h) - 1))

        # Sort coords to ensure x1<=x2, y1<=y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        bbox = [x1, y1, x2, y2]
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        # Convert frame_index from original FPS to 10fps
        frame_index_10fps = int(frame_index / (original_fps / 10.0))
        frame_index_10fps = min(frame_index_10fps, len(frames_10fps) - 1)

        print(f"[SAM2] Tracking from bbox {bbox} on 10fps frame {frame_index_10fps} (original: {frame_index})")

        # Create output masks directory for WSL2 script
        masks_10fps_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks_10fps")
        os.makedirs(masks_10fps_dir, exist_ok=True)

        # Call WSL2 subprocess for SAM2 tracking (PyTorch + torch.compile)
        def _to_wsl_path(p: str) -> str:
            try:
                if len(p) >= 2 and p[1] == ':' and (p[0].isalpha()):
                    drive = p[0].lower()
                    rest = p[2:].replace('\\', '/')
                    rest = rest.lstrip('/')
                    return f"/mnt/{drive}/{rest}"
                return p.replace('\\', '/')
            except Exception:
                return p

        wsl_video = _to_wsl_path(video_10fps_path)
        wsl_masks = _to_wsl_path(masks_10fps_dir)

        wsl_cmd = (
            f'cd /mnt/d/watermarkz && '
            f'source venv_wsl2/bin/activate && '
            f'python sam2_track_wsl2.py "{wsl_video}" "{wsl_masks}" '
            f'--bbox {bbox_str} --frame-idx {frame_index_10fps}'
        )
        print(f"[SAM2-WSL2] Running: wsl -e bash -c \"{wsl_cmd}\"")

        result = subprocess.run(
            ['wsl', '-e', 'bash', '-c', wsl_cmd],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"[SAM2-WSL2] STDERR: {result.stderr}")
            raise RuntimeError(f"WSL2 SAM2 tracking failed: {result.stderr}")

        print(f"[SAM2-WSL2] STDOUT: {result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")

        # Parse result JSON from output
        try:
            # Find the [RESULT] JSON line
            for line in result.stdout.split('\n'):
                if '[RESULT]' in line:
                    json_str = line.split('[RESULT]')[1].strip()
                    sam2_result = json.loads(json_str)
                    print(f"[SAM2-WSL2] Result: {sam2_result}")
                    break
            else:
                print(f"[SAM2-WSL2] Warning: No [RESULT] JSON found in output")
        except Exception as e:
            print(f"[SAM2-WSL2] Warning: Could not parse result JSON: {e}")

        # Read masks from output directory
        mask_files = sorted(glob.glob(os.path.join(masks_10fps_dir, "*.png")))
        if not mask_files:
            raise RuntimeError(f"No masks generated in {masks_10fps_dir}")

        masks_10fps = []
        for mask_file in mask_files:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks_10fps.append(mask)

        print(f"[SAM2-WSL2] Loaded {len(masks_10fps)} masks from {masks_10fps_dir}")

        # --- 6. Extract ALL frames from original video (for ProPainter) ---
        self.update_state(state='PROCESSING', meta={'progress': 15, 'status': 'Extracting full-res frames'})
        print(f"[SAM2] Extracting all {total_frames_original} original frames...")
        cap = cv2.VideoCapture(video_path)
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        total_frames = len(all_frames)
        print(f"[SAM2] Extracted {total_frames} original frames")

        # --- 7. Expand 10fps masks to full FPS (zero-copy in memory) ---
        self.update_state(state='PROCESSING', meta={'progress': 18, 'status': 'Expanding masks to full FPS'})
        print(f"[SAM2] Expanding {len(masks_10fps)} 10fps masks to {total_frames} full-FPS masks...")
        all_masks = expand_masks_10fps(masks_10fps, total_frames, original_fps)
        print(f"[SAM2] Expanded to {len(all_masks)} masks (zero-copy references)")

        # --- 8. Validate masks and count coverage (in memory) ---
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'Validating masks'})

        masks_with_content = sum(1 for m in all_masks if np.sum(m > 127) > 0)
        print(f"[SAM2] Mask validation: {masks_with_content}/{len(all_masks)} frames have mask content")

        if masks_with_content == 0:
            raise RuntimeError("No mask content generated - SAM2 tracking failed")

        # --- 9. Make arrays contiguous for GPU processing ---
        print(f"[SAM2] Preparing arrays for ProPainter...")
        all_frames_contiguous = [np.ascontiguousarray(f) for f in all_frames]
        all_masks_contiguous = [np.ascontiguousarray(m) for m in all_masks]

        # Create output directory for ProPainter
        output_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_output")
        os.makedirs(output_dir, exist_ok=True)

        # --- 10. Run ProPainter with IN-MEMORY arrays (ZERO disk I/O) ---
        print(f"[SAM2] Running ProPainter with TensorRT NeuFlow (in-memory)...")
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': 'Running ProPainter'})

        try:
            faster_propainter_pipeline = get_propainter_pipeline()
            import torch
            use_fp16 = torch.cuda.is_available()

            print(f"[SAM2] ProPainter config:")
            print(f"   - Frames: {len(all_frames_contiguous)} in memory")
            print(f"   - Masks: {len(all_masks_contiguous)} in memory")
            print(f"   - FP16: {use_fp16}")
            print(f"   - Optical Flow: NeuFlow TRT (USE_NEUFLOW={os.getenv('USE_NEUFLOW', '0')})")

            # Run ProPainter with in-memory arrays - SAME as local test script!
            faster_propainter_pipeline(
                video=video_path,  # Used for metadata only
                mask='dummy_mask',  # Not used when masks_array provided
                output=output_dir,
                resize_ratio=1.0,
                mask_dilation=4,
                ref_stride=15,
                neighbor_length=10,
                subvideo_length=120,
                raft_iter=10,
                mode="video_inpainting",
                save_fps=int(original_fps),
                save_frames=True,
                fp16=use_fp16,
                use_cached_models=True,  # Reuse loaded models
                frames_array=all_frames_contiguous,  # Direct memory input (skips disk I/O!)
                masks_array=all_masks_contiguous
            )

            print(f"[SAM2] ProPainter complete!")
        except Exception as e:
            print(f"[ERROR] ProPainter failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        # --- 11. Find ProPainter output video ---
        self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Finding output'})

        propainter_output_video = None
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file == 'inpaint_out.mp4':
                    propainter_output_video = os.path.join(root, file)
                    break
            if propainter_output_video:
                break

        if not propainter_output_video or not os.path.exists(propainter_output_video):
            raise RuntimeError(f"ProPainter output not found in {output_dir}")

        print(f"[SAM2] ProPainter output: {propainter_output_video}")

        # --- 12. Merge audio from original video ---
        print(f"[SAM2] Merging audio from original video...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Encoding final video'})

        output_path = os.path.join(RESULT_DIR, f"{video_id}_sam2_removed.mp4")

        ffmpeg_cmd = [
            str(FFMPEG_EXE),
            '-y',
            '-i', propainter_output_video,
            '-i', video_path,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg encoding failed: {result.stderr}")
            raise RuntimeError(f"Video encoding failed: {result.stderr}")

        print(f"[SAM2] Final video: {output_path}")

        # --- 13. Cleanup temp files ---
        print(f"[SAM2] Cleaning up...")
        for temp_path in [output_dir, video_10fps_path]:
            if isinstance(temp_path, str) and os.path.exists(temp_path):
                if os.path.isdir(temp_path):
                    shutil.rmtree(temp_path)
                else:
                    os.remove(temp_path)

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})

        print(f"[SAM2] Complete! Output: {output_path}")

        return {
            'status': 'success',
            'video_id': video_id,
            'video_path': video_path,
            'output_path': output_path,
            'total_frames': total_frames,
            'masks_generated': len(masks_10fps),
            'masks_expanded': len(all_masks),
            'width': width,
            'height': height,
            'fps': original_fps,
            'pipeline': '10fps_optimized',
            'message': 'SAM2 10fps pipeline complete!'
        }

    except Exception as e:
        print(f"[ERROR] SAM2 task failed: {e}")
        import traceback
        traceback.print_exc()
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


#============================================================================
# FLASK ROUTES
#============================================================================

@app.route('/api/process_sam2', methods=['POST'])
def process_sam2():
    """
    API endpoint to process video with SAM2 masks

    POST data:
    {
        "video_path": "D:\\watermarkz\\videostotrain\\stock2.mp4",
        "masks_folder": "D:\\watermarkz\\temp_sam2_masks",
        "video_id": "stock2"  # Optional
    }
    """
    data = request.get_json()
    video_path = data.get('video_path')
    masks_folder = data.get('masks_folder')
    video_id = data.get('video_id')

    if not video_path or not masks_folder:
        return jsonify({'error': 'Missing video_path or masks_folder'}), 400

    # Queue Celery task
    result = celery.send_task(
        'watermark.process_sam2_interactive',
        args=[video_path, masks_folder, video_id]
    )

    return jsonify({
        'task_id': result.id,
        'status': 'queued',
        'message': 'SAM2 processing task queued'
    })


@app.route('/api/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a Celery task"""
    task = celery.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Pending...'}
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'current': task.info.get('progress', 0),
            'status': task.info.get('status', ''),
            'result': task.info if task.state == 'SUCCESS' else None
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }

    return jsonify(response)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'server': 'SAM2 Production'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
