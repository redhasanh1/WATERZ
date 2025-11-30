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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Segment detection for parallel processing
from segment_detector import detect_segments_from_masks, merge_adjacent_segments, detect_segments
from crop_utils import calculate_crop_region

# Number of parallel segment workers
SEGMENT_WORKERS = int(os.getenv('SEGMENT_WORKERS', '4'))
# Parallel segment execution (inside a single video task)
SAM2_PARALLEL_SEGMENTS = os.getenv('SAM2_PARALLEL_SEGMENTS', '1').lower() in ('1', 'true', 'yes', 'on')

# Temporal padding (in full-FPS frames) around each detected segment
# Helps prevent reappearance at segment boundaries by giving ProPainter context
SEGMENT_TEMPORAL_PAD = int(os.getenv('SEGMENT_TEMPORAL_PAD', '5'))

# Inpainting mask dilation (higher = stronger removal, more context fill)
SAM2_MASK_DILATION = int(os.getenv('SAM2_MASK_DILATION', '4'))
# Strict mode: use single full-video inpainting (no segment split)
SAM2_USE_SEGMENTS = os.getenv('SAM2_USE_SEGMENTS', '0') in ('1', 'true', 'yes', 'on')

# Segment detection sensitivity
# Higher tolerance/gap = less sensitive (fewer segments)
SEGMENT_POS_TOLERANCE = int(os.getenv('SEGMENT_POS_TOLERANCE', '50'))  # match commit defaults
# 10fps domain (legacy fallback)
SEGMENT_MIN_LEN_10FPS = int(os.getenv('SEGMENT_MIN_LEN_10FPS', '10'))
SEGMENT_MERGE_GAP_10FPS = int(os.getenv('SEGMENT_MERGE_GAP_10FPS', '6'))
# Full-FPS domain (preferred)
SEGMENT_MIN_LEN_FULL = int(os.getenv('SEGMENT_MIN_LEN_FULL', '15'))
SEGMENT_MERGE_GAP_FULL = int(os.getenv('SEGMENT_MERGE_GAP_FULL', '60'))
# Detection mode: 'full' (preferred, uses in-memory masks) or '10fps' (dir-based)
SAM2_SEGMENT_DETECTION_MODE = os.getenv('SAM2_SEGMENT_DETECTION_MODE', 'full').strip().lower()

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
# CELERY TASK: SAM2 Interactive Mode - FULL FPS Pipeline
#
# Optimizations:
# - Use original FPS for SAM2 tracking (more accurate masks)
# - Run ProPainter with in-memory arrays (ZERO disk I/O)
# - Use NeuFlow TRT for optical flow (10-70x faster than RAFT)
#============================================================================

@celery.task(bind=True, name='watermark.process_sam2_interactive')
def process_sam2_interactive_task(self, video_path, video_id=None, points=None, video_width=None, video_height=None, frame_index=0, api_base=None):
    """
    SAM2 Interactive Mode - FULL FPS Pipeline:
    - Downloads video from remote
    - Generates masks at original FPS using SAM2Tracker (WSL2)
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
        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Initializing SAM2 FULL-FPS Pipeline'})

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

        # --- 3. Use original FPS for SAM2 (more accurate masks) ---
        self.update_state(state='PROCESSING', meta={'progress': 5, 'status': 'Preparing full-FPS tracking'})
        print(f"[SAM2] Using original FPS for SAM2 tracking (no 10fps conversion)")

        # --- 4. Generate masks at FULL FPS using WSL2 SAM2 subprocess ---
        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Generating masks with SAM2 (FULL FPS via WSL2)'})

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

        # Use original frame index directly (clip to bounds)
        frame_index_full = min(max(int(frame_index), 0), total_frames_original - 1)
        print(f"[SAM2] Tracking from bbox {bbox} on full-FPS frame {frame_index_full}")

        # Create output masks directory for WSL2 script (full FPS)
        masks_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_masks")
        os.makedirs(masks_dir, exist_ok=True)

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

        wsl_video = _to_wsl_path(video_path)
        wsl_masks = _to_wsl_path(masks_dir)

        # Prefer point prompt for higher-quality masks (fallback to bbox via SAM2_PROMPT_MODE=bbox)
        prompt_mode = os.getenv('SAM2_PROMPT_MODE', 'point').strip().lower()
        if prompt_mode == 'point':
            cx = int(np.clip(np.mean(xy[:, 0]), 0, (video_width or width) - 1))
            cy = int(np.clip(np.mean(xy[:, 1]), 0, (video_height or height) - 1))
            prompt_flag = f'--point {cx},{cy}'
            print(f"[SAM2] Using POINT prompt: ({cx},{cy})")
        else:
            prompt_flag = f'--bbox {bbox_str}'
            print(f"[SAM2] Using BBOX prompt: {bbox_str}")

        wsl_cmd = (
            f'cd /mnt/d/watermarkz && '
            f'source venv_wsl2/bin/activate && '
            f'python sam2_track_wsl2.py "{wsl_video}" "{wsl_masks}" '
            f'{prompt_flag} --frame-idx {frame_index_full}'
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

        # Read masks from output directory (FULL FPS)
        mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        if not mask_files:
            raise RuntimeError(f"No masks generated in {masks_dir}")

        masks_full = []
        for mask_file in mask_files:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks_full.append(mask)

        print(f"[SAM2-WSL2] Loaded {len(masks_full)} masks from {masks_dir}")

        # --- 5. Extract ALL frames from original video (for ProPainter) ---
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

        # --- 6. Use FULL-FPS masks directly (no expansion needed) ---
        self.update_state(state='PROCESSING', meta={'progress': 18, 'status': 'Validating masks'})
        if len(masks_full) != total_frames:
            print(f"[SAM2] Warning: mask count {len(masks_full)} != frame count {total_frames}; aligning by truncation")
            min_count = min(len(masks_full), total_frames)
            masks_full = masks_full[:min_count]
            all_frames = all_frames[:min_count]
            total_frames = min_count
        all_masks = masks_full

        # --- 8. Validate masks and count coverage (in memory) ---
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'Validating masks'})

        masks_with_content = sum(1 for m in all_masks if np.sum(m > 127) > 0)
        print(f"[SAM2] Mask validation: {masks_with_content}/{len(all_masks)} frames have mask content")

        if masks_with_content == 0:
            raise RuntimeError("No mask content generated - SAM2 tracking failed")

        # --- 9. Determine segment strategy ---
        if SAM2_USE_SEGMENTS:
            self.update_state(state='PROCESSING', meta={'progress': 22, 'status': 'Detecting segments'})
            if SAM2_SEGMENT_DETECTION_MODE == 'full':
                print(f"[SAM2] Detecting segments from FULL-FPS in-memory masks...")
                # Compute bbox per full-FPS mask in memory, then run detection
                detections_per_frame = []
                for m in all_masks:
                    coords = cv2.findNonZero(m)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        detections_per_frame.append((x, y, x + w, y + h))
                    else:
                        detections_per_frame.append(None)

                segments = detect_segments(
                    detections_per_frame,
                    position_tolerance=SEGMENT_POS_TOLERANCE,
                    min_segment_length=SEGMENT_MIN_LEN_FULL,
                )

                if len(segments) > 1:
                    segments = merge_adjacent_segments(
                        segments,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        max_gap=SEGMENT_MERGE_GAP_FULL,
                    )

                print(f"[SAM2] Detected {len(segments)} segments (FULL-FPS)")
                for idx, (start, end, bbox) in enumerate(segments):
                    print(f"[SAM2]   Segment {idx+1}: frames {start}-{end} ({end-start}f) bbox={bbox}")
            else:
                print(f"[SAM2] Detecting segments from 10fps masks...")

                segments_10fps = detect_segments_from_masks(
                    masks_10fps_dir,
                    position_tolerance=SEGMENT_POS_TOLERANCE,
                    min_segment_length=SEGMENT_MIN_LEN_10FPS,
                    max_segments=80,
                )

                if len(segments_10fps) > 1:
                    segments_10fps = merge_adjacent_segments(
                        segments_10fps,
                        position_tolerance=SEGMENT_POS_TOLERANCE,
                        max_gap=SEGMENT_MERGE_GAP_10FPS,
                    )

                # Scale segments from 10fps to full FPS
                fps_ratio = original_fps / 10.0
                segments = []
                for start_10fps, end_10fps, bbox in segments_10fps:
                    start_full = int(start_10fps * fps_ratio)
                    end_full = min(int((end_10fps + 1) * fps_ratio), total_frames)  # +1 because end is exclusive
                    segments.append((start_full, end_full, bbox))

                print(f"[SAM2] Detected {len(segments)} segments (scaled to full FPS)")
                for idx, (start, end, bbox) in enumerate(segments):
                    print(f"[SAM2]   Segment {idx+1}: frames {start}-{end} ({end-start}f) bbox={bbox}")
        else:
            # Strict monolithic mode (commit logic): process entire video in a single pass
            segments = [(0, total_frames, [0, 0, width, height])]
            print(f"[SAM2] Strict full-video mode: 1 segment covering all frames")

        # --- 10. Calculate global crop region from all masks ---
        print(f"[SAM2] Calculating global crop region...")
        min_x, min_y = width, height
        max_x, max_y = 0, 0

        for mask in all_masks:
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

        if max_x > min_x and max_y > min_y:
            global_bbox = [min_x, min_y, max_x, max_y]
        else:
            global_bbox = [0, 0, width, height]

        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(global_bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"[SAM2] Global crop: {crop_x},{crop_y} {crop_w}x{crop_h}")

        # Create output directory for ProPainter
        output_dir = os.path.join(TEMP_DIR, f"{video_id}_sam2_output")
        os.makedirs(output_dir, exist_ok=True)

        # --- 11. Process segments with ProPainter ---
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': f'Processing {len(segments)} segments'})

        faster_propainter_pipeline = get_propainter_pipeline()
        import torch
        use_fp16 = torch.cuda.is_available()

            print(f"[SAM2] ProPainter config:")
        print(f"   - Segments: {len(segments)}")
        print(f"   - Workers: {SEGMENT_WORKERS}")
        print(f"   - FP16: {use_fp16}")
            print(f"   - Optical Flow: NeuFlow TRT (USE_NEUFLOW={os.getenv('USE_NEUFLOW', '0')})")

        # Crop all frames and masks to global crop region
        all_frames_cropped = [f[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for f in all_frames]
        all_masks_cropped = [m[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] for m in all_masks]

        # Store segment results for merging
        segment_results = {}

        def process_segment(seg_idx, start_f, end_f, seg_bbox):
            """Process a single segment with ProPainter"""
            try:
                duration = end_f - start_f
                # Calculate segment-specific crop within the global crop
                seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                    [seg_bbox[0] - crop_x, seg_bbox[1] - crop_y,
                     seg_bbox[2] - crop_x, seg_bbox[3] - crop_y],
                    crop_w, crop_h, padding_ratio=0.15, min_size=128
                )

                # Determine optimization level based on movement
                movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
                if movement < 10:
                    neighbor_length, subvideo_length = 2, 20
                else:
                    neighbor_length, subvideo_length = 2, 40

                # Temporal padding for context
                pad_left = min(SEGMENT_TEMPORAL_PAD, start_f)
                pad_right = min(SEGMENT_TEMPORAL_PAD, total_frames - end_f)
                proc_start = start_f - pad_left
                proc_end = end_f + pad_right

                print(f"[SAM2] Segment {seg_idx+1}: frames {start_f}-{end_f} ({duration}f), crop={seg_crop_w}x{seg_crop_h}, pad=±{SEGMENT_TEMPORAL_PAD} => proc {proc_start}-{proc_end}")

                # Extract segment frames and masks (double-cropped: global + segment)
                seg_frames = []
                seg_masks = []
                for i in range(proc_start, proc_end):
                    if i < len(all_frames_cropped):
                        seg_frame = all_frames_cropped[i][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                        seg_frames.append(np.ascontiguousarray(seg_frame))
                    if i < len(all_masks_cropped):
                        seg_mask = all_masks_cropped[i][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                        seg_masks.append(np.ascontiguousarray(seg_mask))

                if not seg_frames or not seg_masks:
                    return seg_idx, None, None

                # Create segment output directory
                seg_output_dir = os.path.join(output_dir, f"segment_{seg_idx}")
                os.makedirs(seg_output_dir, exist_ok=True)

                # Run ProPainter on segment
                faster_propainter_pipeline(
                    video='dummy',
                    mask='dummy',
                    output=seg_output_dir,
                    resize_ratio=1.0,
                    mask_dilation=SAM2_MASK_DILATION,
                    ref_stride=15,
                    neighbor_length=neighbor_length,
                    subvideo_length=subvideo_length,
                    raft_iter=10,
                    mode="video_inpainting",
                    save_fps=int(original_fps),
                    save_frames=True,
                    fp16=use_fp16,
                    use_cached_models=True,
                    frames_array=seg_frames,
                    masks_array=seg_masks
                )

                # Load output frames
                output_frames = []
                frames_subdir = None
                for d in os.listdir(seg_output_dir):
                    candidate = os.path.join(seg_output_dir, d, 'frames')
                    if os.path.isdir(candidate):
                        frames_subdir = candidate
                        break

                if frames_subdir:
                    frame_files = sorted([f for f in os.listdir(frames_subdir) if f.endswith('.png')])
                    for ff in frame_files:
                        frame = cv2.imread(os.path.join(frames_subdir, ff))
                        if frame is not None:
                            output_frames.append(frame)

                # Trim padded outputs: keep only frames corresponding to [start_f, end_f)
                # output_frames maps to [proc_start, proc_end); compute slice indices
                keep_start = start_f - proc_start
                keep_end = keep_start + (end_f - start_f)
                if 0 <= keep_start <= len(output_frames) and 0 <= keep_end <= len(output_frames):
                    output_frames = output_frames[keep_start:keep_end]

                return seg_idx, output_frames, (seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h)

            except Exception as e:
                print(f"[ERROR] Segment {seg_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                return seg_idx, None, None

        # Process segments (sequentially for GPU, parallel prep could be added)
        if len(segments) <= 1:
            # Single segment - process entire video as one
            print(f"[SAM2] Processing as single segment...")
            all_frames_contiguous = [np.ascontiguousarray(f) for f in all_frames_cropped]
            all_masks_contiguous = [np.ascontiguousarray(m) for m in all_masks_cropped]

            faster_propainter_pipeline(
                video=video_path,
                mask='dummy_mask',
                output=output_dir,
                resize_ratio=1.0,
                mask_dilation=SAM2_MASK_DILATION,
                ref_stride=15,
                neighbor_length=10,
                subvideo_length=120,
                raft_iter=10,
                mode="video_inpainting",
                save_fps=int(original_fps),
                save_frames=True,
                fp16=use_fp16,
                use_cached_models=True,
                frames_array=all_frames_contiguous,
                masks_array=all_masks_contiguous
            )

            # Find output frames
            final_cropped_frames = []
            for root, dirs, files in os.walk(output_dir):
                if os.path.basename(root) == 'frames':
                    frame_files = sorted([f for f in files if f.endswith('.png')])
                    for ff in frame_files:
                        frame = cv2.imread(os.path.join(root, ff))
                        if frame is not None:
                            final_cropped_frames.append(frame)
                    break

        else:
            # Multiple segments - process each (optionally in parallel) and merge
            if SAM2_PARALLEL_SEGMENTS and SEGMENT_WORKERS > 1:
                print(f"[SAM2] Processing {len(segments)} segments in PARALLEL (workers={min(SEGMENT_WORKERS, len(segments))})...")
                self.update_state(state='PROCESSING', meta={'progress': 30, 'status': f'Processing {len(segments)} segments (parallel)'})

                max_workers = min(SEGMENT_WORKERS, len(segments))
                futures = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                        futures[executor.submit(process_segment, seg_idx, start_f, end_f, seg_bbox)] = seg_idx

                    completed = 0
                    for future in as_completed(futures):
                        seg_idx_done = futures[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            print(f"[ERROR] Segment {seg_idx_done} raised: {e}")
                            result = (seg_idx_done, None, None)
                        segment_results[seg_idx_done] = result
                        completed += 1
                        progress = 30 + int((completed / len(segments)) * 40)
                        self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'Processed {completed}/{len(segments)} segments'})
            else:
                print(f"[SAM2] Processing {len(segments)} segments (sequential)...")
                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                    progress = 30 + int((seg_idx / len(segments)) * 40)
                    self.update_state(state='PROCESSING', meta={
                        'progress': progress,
                        'status': f'Processing segment {seg_idx+1}/{len(segments)}'
                    })

                    clear_gpu_memory()
                    result = process_segment(seg_idx, start_f, end_f, seg_bbox)
                    segment_results[seg_idx] = result

            # --- 12. Merge segments back to cropped frames ---
            print(f"[SAM2] Merging {len(segments)} segments back...")
            self.update_state(state='PROCESSING', meta={'progress': 75, 'status': 'Merging segments'})

            # Start with original cropped frames
            final_cropped_frames = [f.copy() for f in all_frames_cropped]

            for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                if seg_idx not in segment_results:
                    continue
                _, output_frames, crop_info = segment_results[seg_idx]
                if output_frames is None or crop_info is None:
                    continue

                seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = crop_info

                for i, frame_idx in enumerate(range(start_f, end_f)):
                    if frame_idx < len(final_cropped_frames) and i < len(output_frames):
                        # Paste segment result back into cropped frame
                        final_cropped_frames[frame_idx][seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w] = output_frames[i]

        # --- 13. Merge cropped frames back to full resolution ---
        print(f"[SAM2] Merging to full resolution...")
        self.update_state(state='PROCESSING', meta={'progress': 78, 'status': 'Merging to full resolution'})

        final_frames = []
        for i, orig_frame in enumerate(all_frames):
            result_frame = orig_frame.copy()
            if i < len(final_cropped_frames):
                result_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = final_cropped_frames[i]
            final_frames.append(result_frame)

        # --- 14. Encode final video from frames ---
        print(f"[SAM2] Encoding final video ({len(final_frames)} frames)...")
        self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Encoding video'})

        # Save frames temporarily for FFmpeg
        final_frames_dir = os.path.join(output_dir, 'final_frames')
        os.makedirs(final_frames_dir, exist_ok=True)

        for i, frame in enumerate(final_frames):
            cv2.imwrite(os.path.join(final_frames_dir, f'{i:05d}.png'), frame)

        print(f"[SAM2] Saved {len(final_frames)} final frames")

        # --- 15. Encode final video from frames with audio ---
        print(f"[SAM2] Encoding final video with audio...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Encoding final video'})

        output_path = os.path.join(RESULT_DIR, f"{video_id}_sam2_removed.mp4")

        ffmpeg_cmd = [
            str(FFMPEG_EXE),
            '-y',
            '-framerate', str(int(original_fps)),
            '-i', os.path.join(final_frames_dir, '%05d.png'),
            '-i', video_path,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            output_path
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg encoding failed: {result.stderr}")
            raise RuntimeError(f"Video encoding failed: {result.stderr}")

        print(f"[SAM2] Final video: {output_path}")

        # --- 16. Cleanup temp files ---
        print(f"[SAM2] Cleaning up...")
        for temp_path in [output_dir, masks_dir]:
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
            'masks_generated': len(all_masks),
            'masks_expanded': len(all_masks),
            'segments_processed': len(segments),
            'width': width,
            'height': height,
            'fps': original_fps,
            'pipeline': 'full_fps_segments',
            'message': f'SAM2 pipeline complete! Processed {len(segments)} segment(s)'
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
