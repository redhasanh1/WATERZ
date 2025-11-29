"""
SAM2 Watermark Removal Server - Parallel Production System
Isolated from YOLO production (server_production.py)

This server handles SAM2-based watermark removal using pre-generated masks.
"""

import os
import sys
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

# Get FFmpeg/FFprobe executables
def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths"""
    ffmpeg_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffmpeg.exe')
    ffprobe_exe = os.path.join(BASE_DIR, 'ffmpeg', 'ffprobe.exe')

    if not os.path.exists(ffmpeg_exe):
        ffmpeg_exe = 'ffmpeg'
    if not os.path.exists(ffprobe_exe):
        ffprobe_exe = 'ffprobe'

    return ffmpeg_exe, ffprobe_exe

FFMPEG_EXE, FFPROBE_EXE = get_ffmpeg_executables()

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
# CELERY TASK: SAM2 Interactive Mode
# (Copied from commit ff7b669a - lines 1547-2343)
#============================================================================

@celery.task(bind=True, name='watermark.process_sam2_interactive')
def process_sam2_interactive_task(self, video_path, video_id=None):
    """
    SAM2 Interactive Mode: Process video with user-provided SAM2 masks
    - Skip YOLO detection completely
    - Load SAM2 masks from a locally constructed path
    - Run ProPainter with full optimizations
    - Return processed video

    Args:
        video_path: Path to input video
        video_id: Optional video ID for tracking
    """
    try:
        import shutil
        import json
        import numpy as np
        import cv2

        self.update_state(state='STARTED', meta={'progress': 0, 'status': 'Loading SAM2 masks'})

        if not _check_propainter_assets():
            raise RuntimeError("ProPainter assets missing")

        # Generate video ID if not provided
        if not video_id:
            video_id = os.path.basename(video_path).split('.')[0][:8]

        # === FIX: Construct the masks_folder path locally ===
        masks_folder = os.path.join(TEMP_DIR, f"sam2_masks_{video_id}")

        print(f"\n[SAM2 INTERACTIVE] Processing video: {video_path}")
        print(f"[SAM2 INTERACTIVE] Constructed local masks path: {masks_folder}")

        # Check if masks folder exists
        if not os.path.exists(masks_folder):
            raise ValueError(f"Masks folder not found: {masks_folder}")

        # Get video metadata using FFprobe
        try:
            result = subprocess.run([
                str(FFPROBE_EXE),
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
                '-of', 'json',
                video_path
            ], capture_output=True, text=True, timeout=10)

            data = json.loads(result.stdout)
            stream = data['streams'][0]

            width = int(stream['width'])
            height = int(stream['height'])

            # Parse frame rate
            fps_parts = stream['r_frame_rate'].split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1])

            # Get frame count
            total_frames = int(stream.get('nb_frames', 0))
            if total_frames == 0:
                mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.png')])
                total_frames = len(mask_files)
        except Exception as e:
            print(f"[ERROR] Failed to get video metadata: {e}")
            mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.png')])
            total_frames = len(mask_files)
            width = 1920
            height = 1080
            fps = 30.0

        print(f"[SAM2 INTERACTIVE] Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Load SAM2 masks
        mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.png')])
        print(f"[SAM2 INTERACTIVE] Found {len(mask_files)} SAM2 masks")

        if len(mask_files) != total_frames:
            print(f"[WARNING] Mask count mismatch: {len(mask_files)} masks vs {total_frames} frames")

        self.update_state(state='PROCESSING', meta={'progress': 10, 'status': 'Extracting frames'})

        # Create temp directories
        base_name = os.path.basename(video_path).rsplit('.', 1)[0]
        temp_prefix = f"{base_name}_{video_id}_sam2"
        frames_dir = os.path.join(TEMP_DIR, f"{temp_prefix}_frames")
        cropped_dir = os.path.join(TEMP_DIR, f"{temp_prefix}_cropped")
        sam2_masks_dir = os.path.join(TEMP_DIR, f"{temp_prefix}_masks")
        output_dir = os.path.join(TEMP_DIR, f"{temp_prefix}_output")

        for path in [frames_dir, cropped_dir, sam2_masks_dir, output_dir]:
            os.makedirs(path, exist_ok=True)

        # Extract frames
        print(f"[SAM2 INTERACTIVE] Extracting frames from video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frames_dir, f"{frame_idx:04d}.png")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()

        extracted_frames = frame_idx
        print(f"[SAM2 INTERACTIVE] Extracted {extracted_frames} frames")

        # Copy SAM2 masks
        print(f"[SAM2 INTERACTIVE] Loading {len(mask_files)} SAM2 masks...")
        self.update_state(state='PROCESSING', meta={'progress': 20, 'status': 'Loading SAM2 masks'})

        for i, mask_file in enumerate(mask_files):
            src = os.path.join(masks_folder, mask_file)
            dst = os.path.join(sam2_masks_dir, f"{i:04d}.png")
            shutil.copy2(src, dst)

        # Calculate bounding box from ALL masks
        print(f"[SAM2 INTERACTIVE] Analyzing ALL {len(mask_files)} masks to determine crop region...")
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        masks_with_content = 0
        total_white_pixels = 0

        for i, mask_file in enumerate(mask_files):
            mask_path = os.path.join(sam2_masks_dir, f"{i:04d}.png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is not None:
                white_pixels = np.sum(mask > 127)
                if white_pixels > 0:
                    masks_with_content += 1
                    total_white_pixels += white_pixels

                    coords = cv2.findNonZero(mask)
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        min_x = min(min_x, x)
                        min_y = min(min_y, y)
                        max_x = max(max_x, x + w)
                        max_y = max(max_y, y + h)

        # Validate mask coverage
        avg_pixels_per_frame = total_white_pixels / len(mask_files) if len(mask_files) > 0 else 0
        avg_coverage_pct = (avg_pixels_per_frame / (width * height)) * 100

        print(f"[SAM2 INTERACTIVE] Mask validation:")
        print(f"   - Frames with masks: {masks_with_content}/{len(mask_files)}")
        print(f"   - Avg coverage: {avg_coverage_pct:.2f}% per frame")

        if masks_with_content == 0:
            print(f"[WARNING] No mask content found in any frame! Using full frame.")
            bbox = [0, 0, width, height]
        elif masks_with_content < len(mask_files) * 0.5:
            print(f"[WARNING] Only {masks_with_content}/{len(mask_files)} frames have masks")
            bbox = [min_x, min_y, max_x, max_y]
        else:
            bbox = [min_x, min_y, max_x, max_y]
            print(f"[SAM2 INTERACTIVE] Detected union bbox: {bbox}")

        # Calculate crop region
        from crop_utils import calculate_crop_region
        crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
        print(f"[SAM2 INTERACTIVE] Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

        # Crop frames and masks
        print(f"[SAM2 INTERACTIVE] Cropping frames and masks...")
        self.update_state(state='PROCESSING', meta={'progress': 30, 'status': 'Cropping frames'})

        for i in range(extracted_frames):
            frame_file = f"{i:04d}.png"
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                cv2.imwrite(os.path.join(cropped_dir, frame_file), cropped)

        for i in range(len(mask_files)):
            mask_file = f"{i:04d}.png"
            mask_path = os.path.join(sam2_masks_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                cropped_mask = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                cv2.imwrite(mask_path, cropped_mask)

        # Detect segments
        print(f"[SAM2 INTERACTIVE] Detecting segments from masks...")
        from segment_detector import detect_segments_from_masks, merge_adjacent_segments
        position_tolerance = int(os.getenv('SAM2_POSITION_TOLERANCE', '50'))
        min_segment_length = int(os.getenv('SAM2_MIN_SEGMENT_LENGTH', '3'))
        max_segments = int(os.getenv('SAM2_MAX_SEGMENTS', '80'))

        segments = detect_segments_from_masks(
            sam2_masks_dir,
            position_tolerance=position_tolerance,
            min_segment_length=min_segment_length,
            max_segments=max_segments
        )

        if len(segments) > 1:
            segments = merge_adjacent_segments(segments, position_tolerance=position_tolerance, max_gap=60)
            print(f"[SAM2 ADAPTIVE] After merging: {len(segments)} segments")

        # Analyze segment coverage
        frames_with_masks = set()
        for i in range(extracted_frames):
            mask_path = os.path.join(sam2_masks_dir, f"{i:04d}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None and np.count_nonzero(mask) > 0:
                    frames_with_masks.add(i)

        frames_in_segments = set()
        for start_f, end_f, _ in segments:
            frames_in_segments.update(range(start_f, end_f + 1))

        uncovered_frames = sorted(frames_with_masks - frames_in_segments)
        covered_frames = sorted(frames_with_masks & frames_in_segments)

        print(f"[SAM2 COVERAGE] Frames with masks: {len(frames_with_masks)}")
        print(f"[SAM2 COVERAGE] Frames in segments: {len(frames_in_segments)}")
        print(f"[SAM2 COVERAGE] Coverage: {len(covered_frames)}/{len(frames_with_masks)} ({len(covered_frames)/len(frames_with_masks)*100:.1f}%)")

        if uncovered_frames:
            print(f"[SAM2 COVERAGE] Creating filler segments for {len(uncovered_frames)} uncovered frames...")
            filler_count = 0

            for frame_idx in uncovered_frames:
                mask_path = os.path.join(sam2_masks_dir, f"{frame_idx:04d}.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        coords = cv2.findNonZero(mask)
                        if coords is not None:
                            x, y, w, h = cv2.boundingRect(coords)
                            bbox = (x, y, x+w, y+h)

                            start_frame = frame_idx
                            end_frame = frame_idx

                            if frame_idx < extracted_frames - 1 and (frame_idx + 1) not in frames_in_segments:
                                end_frame = frame_idx + 1
                            elif frame_idx > 0 and (frame_idx - 1) not in frames_in_segments:
                                start_frame = frame_idx - 1

                            segments.append((start_frame, end_frame, bbox))
                            filler_count += 1

            segments.sort(key=lambda seg: seg[0])
            print(f"[SAM2 COVERAGE] Created {filler_count} filler segments")
            print(f"[SAM2 COVERAGE] Total segments: {len(segments)}")

        # Run ProPainter
        print(f"[SAM2 ADAPTIVE] Running ProPainter with {len(segments) if segments else 1} segment(s)...")
        self.update_state(state='PROCESSING', meta={'progress': 40, 'status': 'Running ProPainter'})

        try:
            faster_propainter_pipeline = get_propainter_pipeline()
            import torch
            use_fp16 = torch.cuda.is_available()

            if len(segments) == 0 or len(segments) == 1:
                # Single segment
                if len(segments) == 1:
                    start_f, end_f, seg_bbox = segments[0]
                    movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
                    is_stationary = movement < int(os.getenv('SAM2_STATIONARY_THRESHOLD', '10'))
                else:
                    is_stationary = False

                if is_stationary:
                    neighbor_length, subvideo_length = 4, 20
                else:
                    neighbor_length, subvideo_length = 6, 40

                print(f"[SAM2 ADAPTIVE] Processing full video with neighbor={neighbor_length}, subvideo={subvideo_length}")

                faster_propainter_pipeline(
                    video=cropped_dir,
                    mask=sam2_masks_dir,
                    output=output_dir,
                    resize_ratio=1.0,
                    mask_dilation=4,
                    ref_stride=15,
                    neighbor_length=neighbor_length,
                    subvideo_length=subvideo_length,
                    raft_iter=10,
                    mode="video_inpainting",
                    save_frames=True,
                    fp16=use_fp16,
                    frames_array=None,
                    masks_array=None
                )

                video_name = os.path.basename(cropped_dir)
                propainter_output = os.path.join(output_dir, video_name, "frames")

            else:
                # Multiple segments
                print(f"[SAM2 ADAPTIVE] Processing {len(segments)} segments...")

                seg_outputs = []
                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                    seg_output = os.path.join(output_dir, f"segment_{seg_idx}")
                    os.makedirs(seg_output, exist_ok=True)
                    seg_outputs.append(seg_output)

                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                    duration = end_f - start_f + 1
                    movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
                    is_stationary = movement < int(os.getenv('SAM2_STATIONARY_THRESHOLD', '10'))

                    if is_stationary:
                        neighbor_length, subvideo_length = 4, 20
                    else:
                        neighbor_length, subvideo_length = 6, 40

                    print(f"\n[SAM2 ADAPTIVE] Segment {seg_idx+1}/{len(segments)}: frames {start_f}-{end_f} ({duration}f)")
                    print(f"[SAM2 ADAPTIVE]   - neighbor={neighbor_length}, subvideo={subvideo_length}")

                    # Calculate segment crop
                    seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                        seg_bbox, crop_w, crop_h, padding_ratio=0.15, min_size=128
                    )

                    # Extract segment frames/masks
                    seg_frames_dir = os.path.join(output_dir, f"segment_{seg_idx}_frames")
                    seg_masks_dir = os.path.join(output_dir, f"segment_{seg_idx}_masks")
                    os.makedirs(seg_frames_dir, exist_ok=True)
                    os.makedirs(seg_masks_dir, exist_ok=True)

                    for frame_idx in range(start_f, end_f + 1):
                        frame_file = f"{frame_idx:04d}.png"
                        src_frame = os.path.join(cropped_dir, frame_file)
                        src_mask = os.path.join(sam2_masks_dir, frame_file)

                        if os.path.exists(src_frame):
                            frame = cv2.imread(src_frame)
                            seg_frame = frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                            dst_frame = os.path.join(seg_frames_dir, f"{frame_idx-start_f:04d}.png")
                            cv2.imwrite(dst_frame, seg_frame)

                        if os.path.exists(src_mask):
                            mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)
                            seg_mask = mask[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                        else:
                            seg_mask = np.zeros((seg_crop_h, seg_crop_w), dtype=np.uint8)

                        dst_mask = os.path.join(seg_masks_dir, f"{frame_idx-start_f:04d}.png")
                        cv2.imwrite(dst_mask, seg_mask)

                    # Process segment
                    faster_propainter_pipeline(
                        video=seg_frames_dir,
                        mask=seg_masks_dir,
                        output=seg_outputs[seg_idx],
                        resize_ratio=1.0,
                        mask_dilation=4,
                        ref_stride=15,
                        neighbor_length=neighbor_length,
                        subvideo_length=subvideo_length,
                        raft_iter=10,
                        mode="video_inpainting",
                        save_frames=True,
                        fp16=use_fp16,
                        frames_array=None,
                        masks_array=None
                    )

                    print(f"[SAM2 ADAPTIVE] Segment {seg_idx+1} complete!")
                    clear_gpu_memory()

                # Merge segments
                print(f"[SAM2 ADAPTIVE] Merging {len(segments)} segments...")

                merged_output = os.path.join(output_dir, "merged_frames")
                os.makedirs(merged_output, exist_ok=True)

                # Copy all original frames
                for i in range(extracted_frames):
                    src_frame = os.path.join(cropped_dir, f"{i:04d}.png")
                    dst_frame = os.path.join(merged_output, f"{i:04d}.png")
                    if os.path.exists(src_frame):
                        shutil.copy2(src_frame, dst_frame)

                # Paste cleaned segments
                for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                    seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                        seg_bbox, crop_w, crop_h, padding_ratio=0.15, min_size=128
                    )

                    video_dirname = f"segment_{seg_idx}_frames"
                    seg_propainter_output = os.path.join(seg_outputs[seg_idx], video_dirname, "frames")

                    if not os.path.exists(seg_propainter_output):
                        print(f"[ERROR] Segment {seg_idx} output not found: {seg_propainter_output}")
                        continue

                    for frame_idx in range(start_f, end_f + 1):
                        src_file = os.path.join(seg_propainter_output, f"{frame_idx-start_f:04d}.png")

                        if not os.path.exists(src_file):
                            continue

                        cleaned_seg = cv2.imread(src_file)
                        if cleaned_seg is None:
                            continue

                        dst_file = os.path.join(merged_output, f"{frame_idx:04d}.png")
                        full_frame = cv2.imread(dst_file)

                        if full_frame is None:
                            continue

                        if cleaned_seg.shape != (seg_crop_h, seg_crop_w, 3):
                            cleaned_seg = cv2.resize(cleaned_seg, (seg_crop_w, seg_crop_h), interpolation=cv2.INTER_LINEAR)

                        try:
                            full_frame[seg_crop_y:seg_crop_y+seg_crop_h,
                                     seg_crop_x:seg_crop_x+seg_crop_w] = cleaned_seg
                            cv2.imwrite(dst_file, full_frame)
                        except Exception as e:
                            print(f"[ERROR] Segment {seg_idx} frame {frame_idx}: {e}")

                propainter_output = merged_output

            print(f"[SAM2 INTERACTIVE] ProPainter complete!")
        except Exception as e:
            print(f"[ERROR] ProPainter failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Merge back into original frames
        print(f"[SAM2 INTERACTIVE] Merging back into original frames...")
        self.update_state(state='PROCESSING', meta={'progress': 80, 'status': 'Merging results'})

        if not os.path.exists(propainter_output):
            propainter_output = output_dir

        final_frames_dir = os.path.join(TEMP_DIR, f"{temp_prefix}_final")
        os.makedirs(final_frames_dir, exist_ok=True)

        frames_merged = 0
        for i in range(extracted_frames):
            frame_file = f"{i:04d}.png"

            orig_frame = cv2.imread(os.path.join(frames_dir, frame_file))
            if orig_frame is None:
                continue

            cleaned_path = os.path.join(propainter_output, frame_file)
            if os.path.exists(cleaned_path):
                cleaned_crop = cv2.imread(cleaned_path)
                if cleaned_crop is not None:
                    orig_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop
                    frames_merged += 1

            cv2.imwrite(os.path.join(final_frames_dir, frame_file), orig_frame)

        print(f"[SAM2 INTERACTIVE] Merged {frames_merged}/{extracted_frames} frames")

        # Encode final video
        print(f"[SAM2 INTERACTIVE] Encoding final video...")
        self.update_state(state='PROCESSING', meta={'progress': 90, 'status': 'Encoding video'})

        output_path = os.path.join(RESULT_DIR, f"{video_id}_sam2_removed.mp4")

        ffmpeg_cmd = [
            str(FFMPEG_EXE),
            '-framerate', str(fps),
            '-i', os.path.join(final_frames_dir, '%04d.png'),
            '-i', video_path,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-y',
            output_path
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg encoding failed: {result.stderr}")
            raise RuntimeError(f"Video encoding failed: {result.stderr}")

        print(f"[SAM2 INTERACTIVE] Video encoded: {output_path}")

        # Cleanup
        print(f"[SAM2 INTERACTIVE] Cleaning up...")
        for temp_path in [frames_dir, cropped_dir, sam2_masks_dir, output_dir, final_frames_dir]:
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)

        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Complete'})

        print(f"[SAM2 INTERACTIVE] Complete! Output: {output_path}")

        return {
            'status': 'success',
            'video_id': video_id,
            'video_path': video_path,
            'output_path': output_path,
            'masks_folder': masks_folder,
            'total_frames': extracted_frames,
            'width': width,
            'height': height,
            'fps': fps,
            'message': 'SAM2 processing complete!'
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
