"""
SAM2 LOCAL - Simple Standalone Watermark Removal
No Celery, No Redis, No Server - Just run it!

Usage:
    python RUN_SAM2_LOCAL.py

1. GUI file picker opens
2. Select video
3. Processing starts (uses temp_sam2_masks/)
4. Output saved to results/
"""

import os
import sys
import cv2
import numpy as np
import shutil
import subprocess
import json
import torch
import torch._inductor
import platform
from pathlib import Path
try:
    from tkinter import Tk, filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Global CUDA graphs kill switch (RTX 3070 compatibility)
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.cudagraph_trees = False

# Add ProPainter to path
BASE_DIR = Path(__file__).parent
PROPAINTER_DIR = BASE_DIR / "faster-propainter-main"
sys.path.insert(0, str(PROPAINTER_DIR))

# torch.compile requires Visual Studio C++ environment on Windows
# Use START_CELERY_TRT_LOCAL.bat or call vcvars64.bat first to enable it
IS_WSL2 = 'microsoft' in platform.release().lower() if hasattr(platform, 'release') else False
IS_LINUX = sys.platform.startswith('linux')

# Check if torch.compile was enabled externally (e.g., by batch file)
if os.getenv('USE_TORCH_COMPILE_RAFT', '0') == '1':
    print(f"[COMPILE] torch.compile enabled for RAFT (set externally)")
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(BASE_DIR / '.torch_compile_cache')
    os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
    os.environ['TRITON_CACHE_DIR'] = str(BASE_DIR / '.triton_cache')
    os.environ['TORCH_CUDAGRAPHS'] = '0'
    os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'
elif IS_WSL2 or IS_LINUX:
    # Only auto-enable if user hasn't explicitly disabled it
    if os.getenv('USE_TORCH_COMPILE_RAFT') != '0':
        print(f"[OPTIMIZE] Enabling Flash Attention (Linux/WSL2)")
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(BASE_DIR / '.torch_compile_cache')
        os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
        os.environ['TRITON_CACHE_DIR'] = str(BASE_DIR / '.triton_cache')
        os.environ['USE_TORCH_COMPILE_RAFT'] = '0'  # Disabled - slow on RTX 3070
        os.environ['USE_TORCH_COMPILE'] = '0'  # Disabled - slow on RTX 3070
        os.environ['ENABLE_FLASH_ATTENTION'] = '0'  # Disabled - not supported on RTX 3070
        os.environ['TORCH_CUDAGRAPHS'] = '0'
        os.environ['TORCHINDUCTOR_CUDAGRAPHS'] = '0'
    else:
        print(f"[INFO] torch.compile disabled by user (USE_TORCH_COMPILE_RAFT=0)")
else:
    # Windows: Disable torch.compile by default (requires Visual Studio C++ environment)
    print(f"[INFO] torch.compile disabled on Windows (use START_CELERY_TRT_LOCAL.bat for 1.5x speedup)")
    os.environ['USE_TORCH_COMPILE_RAFT'] = '0'

from watermark import pipeline as faster_propainter_pipeline
from crop_utils import calculate_crop_region
from segment_detector import detect_segments, merge_adjacent_segments
from yolo_detector import YOLOWatermarkDetector

# Paths
TEMP_DIR = BASE_DIR / "temp"
RESULT_DIR = BASE_DIR / "results"
MASKS_FOLDER = BASE_DIR / "temp_sam2_masks"

# FFmpeg
def get_ffmpeg_executables():
    """Get FFmpeg and FFprobe paths with fallback to static-ffmpeg."""
    import platform

    # On Linux, prefer NVENC-capable ffmpeg first
    if platform.system() == 'Linux':
        nvenc_ffmpeg = '/tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg'
        nvenc_ffprobe = '/tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe'
        if os.path.exists(nvenc_ffmpeg):
            print(f"[OK] Using NVENC FFmpeg: {nvenc_ffmpeg}")
            return nvenc_ffmpeg, nvenc_ffprobe

    # Try system PATH first
    ffmpeg_path = shutil.which('ffmpeg')
    ffprobe_path = shutil.which('ffprobe')

    if ffmpeg_path and ffprobe_path:
        print(f"[OK] Using system FFmpeg: {ffmpeg_path}")
        return ffmpeg_path, ffprobe_path

    # Fallback to static-ffmpeg
    try:
        from static_ffmpeg import run
        ffmpeg_path, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
        print(f"[OK] Using static-ffmpeg: {ffmpeg_path}")
        return ffmpeg_path, ffprobe_path
    except ImportError:
        raise RuntimeError("FFmpeg not available. Install via: pip install static-ffmpeg")

FFMPEG_EXE, FFPROBE_EXE = get_ffmpeg_executables()


def get_dynamic_subvideo_length(width, height):
    """Adapt chunk size to video resolution for memory efficiency"""
    resolution = width * height

    if resolution <= 640 * 480:        # 480p
        return 100
    elif resolution <= 1280 * 720:     # 720p
        return 80
    elif resolution <= 1920 * 1080:    # 1080p
        return 60
    elif resolution <= 2560 * 1440:    # 1440p
        return 40
    else:                              # 4K+
        return 20


def select_video():
    """Open GUI file picker to select video"""
    print("\n[GUI] Opening file picker...")

    # Hide root window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Open file dialog
    video_path = filedialog.askopenfilename(
        title="Select Video to Process with SAM2",
        initialdir=str(BASE_DIR / "videostotrain"),
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not video_path:
        print("[CANCELLED] No video selected")
        return None

    return video_path


def get_video_metadata(video_path):
    """Get video metadata using FFprobe"""
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

        return width, height, fps, total_frames
    except Exception as e:
        print(f"[WARNING] FFprobe failed: {e}")
        return None, None, None, None


def process_sam2_local(video_path, masks_folder):
    """
    Process video with SAM2 masks locally (no Celery)
    """
    print("\n" + "="*80)
    print("SAM2 LOCAL WATERMARK REMOVAL")
    print("="*80)

    print(f"\n[OK] Video: {video_path}")

    # Get video metadata
    width, height, fps, total_frames = get_video_metadata(video_path)

    if width is None:
        width, height, fps = 1920, 1080, 30.0
        print(f"[WARNING] Could not get video metadata, using defaults")

    print(f"[OK] Video: {width}x{height} @ {fps:.1f}fps")

    # Generate video ID
    video_id = Path(video_path).stem[:8]

    # Create temp directories
    temp_prefix = f"{video_id}_sam2_local"
    frames_dir = TEMP_DIR / f"{temp_prefix}_frames"
    cropped_dir = TEMP_DIR / f"{temp_prefix}_cropped"
    sam2_masks_dir = TEMP_DIR / f"{temp_prefix}_masks"
    output_dir = TEMP_DIR / f"{temp_prefix}_output"

    # Clean any existing temp directories first (remove stale files from previous runs)
    for path in [frames_dir, cropped_dir, sam2_masks_dir, output_dir]:
        if path.exists():
            print(f"[CLEANUP] Removing stale directory: {path.name}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/7] Extracting frames...")

    # Try NVDEC hardware decoder first, fallback to CPU
    use_nvdec = True
    nvdec_loader = None
    cap = None

    if use_nvdec:
        try:
            from nvdec_video_loader import NVDECVideoLoader
            nvdec_loader = NVDECVideoLoader(str(video_path), device_id=0)
            props = nvdec_loader.get_properties()
            fps = int(props['fps'])
            total_frames = props['total_frames']
            print(f"[OK] NVDEC hardware decoder ready ({total_frames} frames)")
        except ImportError:
            print("[WARNING] nvdec_video_loader not available, falling back to CPU decoder")
            use_nvdec = False

    if not use_nvdec:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return None
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        print(f"[OK] CPU decoder ready ({total_frames} frames)")

    # Decode frames
    import time
    decode_start = time.time()

    if use_nvdec:
        all_frames = nvdec_loader.load_all_frames(to_numpy=True, color_format='BGR')
        extracted_frames = len(all_frames)
        nvdec_loader.close()

        # Uniqueness check - verify frames are different
        if extracted_frames > 1:
            import hashlib
            h0 = hashlib.md5(all_frames[0].tobytes()).hexdigest()[:8]
            h1 = hashlib.md5(all_frames[-1].tobytes()).hexdigest()[:8]
            if h0 == h1:
                raise RuntimeError(f"[NVDEC ERROR] All frames identical (hash={h0})")
            print(f"[OK] NVDEC frames verified unique: first={h0}, last={h1}")

        # Write to disk
        for idx, frame in enumerate(all_frames):
            cv2.imwrite(str(frames_dir / f"{idx:04d}.png"), frame)

        decode_time = time.time() - decode_start
        print(f"[OK] NVDEC decoded {extracted_frames} frames: {decode_time:.3f}s ({decode_time/extracted_frames*1000:.2f}ms/frame)")
    else:
        all_frames = []  # FIX: Initialize all_frames for CPU decoder path
        extracted_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)  # FIX: Collect frames for YOLO detection
            cv2.imwrite(str(frames_dir / f"{extracted_frames:04d}.png"), frame)
            extracted_frames += 1
        cap.release()

        decode_time = time.time() - decode_start
        print(f"[OK] CPU decoded {extracted_frames} frames: {decode_time:.3f}s ({decode_time/extracted_frames*1000:.2f}ms/frame)")

    if extracted_frames == 0:
        print(f"[ERROR] No frames extracted!")
        return None

    # Initialize YOLO detector
    print(f"\n[2/7] Detecting watermarks with YOLO...")
    detector = YOLOWatermarkDetector()

    # CRITICAL: Warmup TensorRT engine
    print(f"[YOLO] Warming up TensorRT engine...")
    dummy_batch = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(64)]
    _ = detector.detect_batch(dummy_batch, confidence_threshold=0.15, batch_size=64)
    print(f"[OK] YOLO detector ready!")

    # Detection parameters (from working server)
    det_conf = 0.15
    feather_pixels = 21

    # Generate masks with YOLO batch detection (server approach)
    print(f"[YOLO] Batch detecting watermarks (conf={det_conf}, batch_size=64)...")

    # Use batch detection on ALL frames at once (proper 640x640 padding)
    all_detections = detector.detect_batch(all_frames, confidence_threshold=det_conf, padding=0, batch_size=64)

    # Count frames with watermarks
    frames_with_watermark = sum(1 for dets in all_detections if dets and len(dets) > 0)
    print(f"[OK] Detected watermarks in {frames_with_watermark}/{extracted_frames} frames")

    # Generate masks from batch detections
    zero_mask = np.zeros((all_frames[0].shape[0], all_frames[0].shape[1]), dtype=np.uint8)

    for i in range(extracted_frames):
        frame = all_frames[i]
        detections = all_detections[i] if i < len(all_detections) else []

        # Generate mask
        if detections and len(detections) > 0:
            mask = detector.create_mask(frame, detections,
                                      expand_ratio=0.0,
                                      expand_pixels=0,
                                      feather_pixels=feather_pixels)
        else:
            mask = zero_mask.copy()

        cv2.imwrite(str(sam2_masks_dir / f"{i:04d}.png"), mask)

    print(f"[OK] Generated {extracted_frames} masks from batch detection")

    if frames_with_watermark == 0:
        print(f"\n[WARNING] No watermark detected in video!")
        print(f"[HINT] Try lowering confidence threshold or check YOLO model")

    print(f"\n[3/7] Analyzing masks to find watermark region...")
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    masks_with_content = 0

    for i in range(extracted_frames):
        mask_path = sam2_masks_dir / f"{i:04d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is not None:
            white_pixels = np.sum(mask > 127)
            if white_pixels > 0:
                masks_with_content += 1
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x + w)
                    max_y = max(max_y, y + h)

    print(f"[OK] Frames with masks: {masks_with_content}/{extracted_frames}")

    if masks_with_content == 0:
        print(f"[WARNING] No mask content found!")
        bbox = [0, 0, width, height]
    else:
        bbox = [min_x, min_y, max_x, max_y]
        print(f"[OK] Watermark bbox: {bbox}")

    # Calculate crop region
    crop_x, crop_y, crop_w, crop_h = calculate_crop_region(bbox, width, height, padding_ratio=0.2, min_size=128)
    print(f"[OK] Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

    print(f"\n[4/7] Cropping frames and masks...")

    # Validate crop region against actual frame dimensions
    sample_frame = cv2.imread(str(frames_dir / "0000.png"))
    if sample_frame is not None:
        actual_h, actual_w = sample_frame.shape[:2]
        print(f"[DEBUG] Actual frame dimensions: {actual_w}x{actual_h}")
        print(f"[DEBUG] Expected frame dimensions: {width}x{height}")

        # Check if frames were resized - need to scale bbox and crop coords
        if actual_w != width or actual_h != height:
            print(f"[WARNING] Frame size mismatch! Frames were resized from {width}x{height} to {actual_w}x{actual_h}")

            # Scale bbox to match actual frame size
            scale_x = actual_w / width
            scale_y = actual_h / height

            # Recalculate crop with scaled dimensions
            scaled_bbox = [
                int(bbox[0] * scale_x),
                int(bbox[1] * scale_y),
                int(bbox[2] * scale_x),
                int(bbox[3] * scale_y)
            ]
            print(f"[OK] Scaled bbox: {scaled_bbox}")

            crop_x, crop_y, crop_w, crop_h = calculate_crop_region(
                scaled_bbox, actual_w, actual_h, padding_ratio=0.2, min_size=128
            )
            print(f"[OK] Recalculated crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

        # Clamp crop region to actual frame bounds
        crop_x = max(0, min(crop_x, actual_w - 1))
        crop_y = max(0, min(crop_y, actual_h - 1))
        crop_w = min(crop_w, actual_w - crop_x)
        crop_h = min(crop_h, actual_h - crop_y)

        # Ensure minimum size for ProPainter (160x160)
        if crop_w < 160 or crop_h < 160:
            print(f"[WARNING] Crop too small ({crop_w}x{crop_h}), using full frame instead")
            crop_x, crop_y = 0, 0
            crop_w, crop_h = actual_w, actual_h

        print(f"[OK] Final crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")

    for i in range(extracted_frames):
        frame_file = f"{i:04d}.png"
        frame = cv2.imread(str(frames_dir / frame_file))
        if frame is not None:
            # Ensure crop is within bounds
            h, w = frame.shape[:2]
            safe_crop_x = min(crop_x, w - 1)
            safe_crop_y = min(crop_y, h - 1)
            safe_crop_w = min(crop_w, w - safe_crop_x)
            safe_crop_h = min(crop_h, h - safe_crop_y)

            cropped = frame[safe_crop_y:safe_crop_y+safe_crop_h, safe_crop_x:safe_crop_x+safe_crop_w]
            if cropped.size > 0:
                cv2.imwrite(str(cropped_dir / frame_file), cropped)
            else:
                print(f"[ERROR] Frame {i} crop resulted in empty image, copying full frame")
                cv2.imwrite(str(cropped_dir / frame_file), frame)

    for i in range(extracted_frames):
        mask_file = f"{i:04d}.png"
        mask = cv2.imread(str(sam2_masks_dir / mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            h, w = mask.shape[:2]
            safe_crop_x = min(crop_x, w - 1)
            safe_crop_y = min(crop_y, h - 1)
            safe_crop_w = min(crop_w, w - safe_crop_x)
            safe_crop_h = min(crop_h, h - safe_crop_y)

            cropped_mask = mask[safe_crop_y:safe_crop_y+safe_crop_h, safe_crop_x:safe_crop_x+safe_crop_w]
            if cropped_mask.size > 0:
                cv2.imwrite(str(sam2_masks_dir / mask_file), cropped_mask)
            else:
                print(f"[ERROR] Mask {i} crop resulted in empty image, copying full mask")
                cv2.imwrite(str(sam2_masks_dir / mask_file), mask)

    print(f"\n[5/7] Detecting motion segments...")
    # Very high tolerance = only split when watermark REALLY moves (150+ pixels)
    position_tolerance = int(os.getenv('SAM2_POSITION_TOLERANCE', '150'))
    # Minimum 60 frames (~2 sec) before creating new segment
    min_segment_length = int(os.getenv('SAM2_MIN_SEGMENT_LENGTH', '60'))

    # Build detections list from masks
    detections_per_frame = []
    for i in range(extracted_frames):
        mask_path = sam2_masks_dir / f"{i:04d}.png"
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                coords = cv2.findNonZero(mask)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    detections_per_frame.append((x, y, x+w, y+h))
                else:
                    detections_per_frame.append(None)
            else:
                detections_per_frame.append(None)
        else:
            detections_per_frame.append(None)

    segments = detect_segments(
        detections_per_frame,
        position_tolerance=position_tolerance,
        min_segment_length=min_segment_length
    )

    if len(segments) > 1:
        segments = merge_adjacent_segments(segments, position_tolerance=position_tolerance, max_gap=60)

    print(f"[OK] Detected {len(segments)} segments")

    # Extend segments to cover all frames (no gap filling, just extend boundaries)
    if segments:
        # Make sure first segment starts at 0
        first_start, first_end, first_bbox = segments[0]
        if first_start > 0:
            segments[0] = (0, first_end, first_bbox)

        # Make sure last segment ends at last frame
        last_start, last_end, last_bbox = segments[-1]
        if last_end < extracted_frames - 1:
            segments[-1] = (last_start, extracted_frames - 1, last_bbox)

        # Fill gaps between segments by extending adjacent segments
        new_segments = []
        for i, (start, end, bbox) in enumerate(segments):
            if i > 0:
                prev_end = new_segments[-1][1]
                if start > prev_end + 1:
                    # Gap exists - extend current segment backward to cover it
                    start = prev_end + 1
            new_segments.append((start, end, bbox))
        segments = new_segments

    print(f"\n[6/7] Running ProPainter on {len(segments) if segments else 1} segment(s)...")

    try:
        import torch
        use_fp16 = torch.cuda.is_available()

        if len(segments) == 0 or len(segments) == 1:
            # Single segment
            if len(segments) == 1:
                start_f, end_f, seg_bbox = segments[0]
                movement = max(seg_bbox[2] - seg_bbox[0], seg_bbox[3] - seg_bbox[1])
                is_stationary = movement < 10
            else:
                is_stationary = False

            # Use server parameters for optimal quality
            neighbor_length = 10
            ref_stride = 10
            dynamic_subvideo = get_dynamic_subvideo_length(width, height)

            print(f"[OK] Processing entire video (neighbor={neighbor_length}, ref_stride={ref_stride}, subvideo={dynamic_subvideo})")

            faster_propainter_pipeline(
                video=str(cropped_dir),
                mask=str(sam2_masks_dir),
                output=str(output_dir),
                resize_ratio=1.0,
                mask_dilation=4,
                ref_stride=ref_stride,
                neighbor_length=neighbor_length,
                subvideo_length=dynamic_subvideo,
                raft_iter=10,
                mode="video_inpainting",
                save_frames=True,
                fp16=use_fp16,
                frames_array=None,
                masks_array=None
            )

            propainter_output = output_dir / cropped_dir.name / "frames"

        else:
            # Multiple segments
            print(f"[OK] Processing {len(segments)} segments with adaptive params...")

            seg_outputs = []
            for seg_idx in range(len(segments)):
                seg_output = output_dir / f"segment_{seg_idx}"
                seg_output.mkdir(exist_ok=True)
                seg_outputs.append(seg_output)

            # Use server parameters for all segments
            neighbor_length = 10
            ref_stride = 10
            dynamic_subvideo = 120  # For cropped segments, use larger chunks

            for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                print(f"  Segment {seg_idx+1}/{len(segments)}: frames {start_f}-{end_f} (neighbor={neighbor_length})")

                seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                    seg_bbox, crop_w, crop_h, padding_ratio=0.15, min_size=128
                )

                seg_frames_dir = output_dir / f"segment_{seg_idx}_frames"
                seg_masks_dir = output_dir / f"segment_{seg_idx}_masks"
                seg_frames_dir.mkdir(exist_ok=True)
                seg_masks_dir.mkdir(exist_ok=True)

                for frame_idx in range(start_f, end_f + 1):
                    frame_file = f"{frame_idx:04d}.png"
                    src_frame = cropped_dir / frame_file
                    src_mask = sam2_masks_dir / frame_file

                    if src_frame.exists():
                        frame = cv2.imread(str(src_frame))
                        seg_frame = frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                        cv2.imwrite(str(seg_frames_dir / f"{frame_idx-start_f:04d}.png"), seg_frame)

                    if src_mask.exists():
                        mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
                        seg_mask = mask[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w]
                    else:
                        seg_mask = np.zeros((seg_crop_h, seg_crop_w), dtype=np.uint8)

                    cv2.imwrite(str(seg_masks_dir / f"{frame_idx-start_f:04d}.png"), seg_mask)

                faster_propainter_pipeline(
                    video=str(seg_frames_dir),
                    mask=str(seg_masks_dir),
                    output=str(seg_outputs[seg_idx]),
                    resize_ratio=1.0,
                    mask_dilation=4,
                    ref_stride=ref_stride,
                    neighbor_length=neighbor_length,
                    subvideo_length=dynamic_subvideo,
                    raft_iter=10,
                    mode="video_inpainting",
                    save_frames=True,
                    fp16=use_fp16,
                    frames_array=None,
                    masks_array=None
                )

                # Clear GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Merge segments
            print(f"[OK] Merging {len(segments)} segments...")

            merged_output = output_dir / "merged_frames"
            merged_output.mkdir(exist_ok=True)

            # Copy all original frames
            for i in range(extracted_frames):
                src = cropped_dir / f"{i:04d}.png"
                dst = merged_output / f"{i:04d}.png"
                if src.exists():
                    shutil.copy2(src, dst)

            # Paste cleaned segments
            for seg_idx, (start_f, end_f, seg_bbox) in enumerate(segments):
                seg_crop_x, seg_crop_y, seg_crop_w, seg_crop_h = calculate_crop_region(
                    seg_bbox, crop_w, crop_h, padding_ratio=0.15, min_size=128
                )

                seg_propainter_output = seg_outputs[seg_idx] / f"segment_{seg_idx}_frames" / "frames"

                if not seg_propainter_output.exists():
                    continue

                for frame_idx in range(start_f, end_f + 1):
                    src_file = seg_propainter_output / f"{frame_idx-start_f:04d}.png"
                    if not src_file.exists():
                        continue

                    cleaned_seg = cv2.imread(str(src_file))
                    if cleaned_seg is None:
                        continue

                    dst_file = merged_output / f"{frame_idx:04d}.png"
                    full_frame = cv2.imread(str(dst_file))
                    if full_frame is None:
                        continue

                    if cleaned_seg.shape != (seg_crop_h, seg_crop_w, 3):
                        cleaned_seg = cv2.resize(cleaned_seg, (seg_crop_w, seg_crop_h))

                    full_frame[seg_crop_y:seg_crop_y+seg_crop_h, seg_crop_x:seg_crop_x+seg_crop_w] = cleaned_seg
                    cv2.imwrite(str(dst_file), full_frame)

            propainter_output = merged_output

        print(f"[OK] ProPainter complete!")

    except Exception as e:
        print(f"\n[ERROR] ProPainter failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f"\n[7/7] Merging back into original frames and encoding...")

    final_frames_dir = TEMP_DIR / f"{temp_prefix}_final"
    final_frames_dir.mkdir(exist_ok=True)

    for i in range(extracted_frames):
        frame_file = f"{i:04d}.png"
        orig_frame = cv2.imread(str(frames_dir / frame_file))

        if orig_frame is not None:
            cleaned_path = propainter_output / frame_file
            if cleaned_path.exists():
                cleaned_crop = cv2.imread(str(cleaned_path))
                if cleaned_crop is not None:
                    orig_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop

            cv2.imwrite(str(final_frames_dir / frame_file), orig_frame)

    # Encode video with audio stream copy (server approach)
    output_path = RESULT_DIR / f"{video_id}_sam2_removed.mp4"
    RESULT_DIR.mkdir(exist_ok=True)

    # Check if original video has audio
    check_audio_cmd = [
        str(FFPROBE_EXE),
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]

    has_audio_result = subprocess.run(check_audio_cmd, capture_output=True, text=True, timeout=10)
    has_audio = 'audio' in has_audio_result.stdout

    if has_audio:
        # Stream copy audio - INSTANT, no quality loss!
        print(f"[OK] Audio detected - using stream copy")
        ffmpeg_cmd = [
            str(FFMPEG_EXE), '-y',
            '-framerate', str(fps),
            '-i', str(final_frames_dir / '%04d.png'),
            '-i', video_path,
            '-map', '0:v:0',       # Video from processed frames
            '-map', '1:a:0',       # Audio from original
            '-c:v', 'h264_nvenc',  # GPU encode (10-20x faster)
            '-preset', 'p1',       # Fastest preset
            '-tune', 'hq',         # High quality tuning
            '-rc', 'vbr',          # Variable bitrate
            '-cq', '19',           # Quality level
            '-b:v', '0',           # Auto bitrate
            '-spatial-aq', '1',    # Spatial AQ
            '-temporal-aq', '1',   # Temporal AQ
            '-c:a', 'copy',        # DON'T re-encode audio (instant!)
            '-pix_fmt', 'yuv420p',
            '-shortest',
            str(output_path)
        ]
    else:
        # No audio, just encode video
        print(f"[OK] No audio detected - video only")
        ffmpeg_cmd = [
            str(FFMPEG_EXE), '-y',
            '-framerate', str(fps),
            '-i', str(final_frames_dir / '%04d.png'),
            '-c:v', 'h264_nvenc',  # GPU encode (10-20x faster)
            '-preset', 'p1',       # Fastest preset
            '-tune', 'hq',         # High quality tuning
            '-rc', 'vbr',          # Variable bitrate
            '-cq', '19',           # Quality level
            '-b:v', '0',           # Auto bitrate
            '-spatial-aq', '1',    # Spatial AQ
            '-temporal-aq', '1',   # Temporal AQ
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]

    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # NVENC failed - fall back to libx264 (software encoding)
        print(f"[WARN] NVENC failed, falling back to libx264 (software encoding)...")

        if has_audio:
            ffmpeg_cmd = [
                str(FFMPEG_EXE), '-y',
                '-framerate', str(fps),
                '-i', str(final_frames_dir / '%04d.png'),
                '-i', video_path,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-c:a', 'copy',
                '-pix_fmt', 'yuv420p',
                '-shortest',
                str(output_path)
            ]
        else:
            ffmpeg_cmd = [
                str(FFMPEG_EXE), '-y',
                '-framerate', str(fps),
                '-i', str(final_frames_dir / '%04d.png'),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                str(output_path)
            ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg encoding failed: {result.stderr}")
            return None
        print(f"[OK] Video encoded with libx264!")
    else:
        print(f"[OK] Video encoded with NVENC!")

    # Cleanup
    print(f"\n[CLEANUP] Removing temp files...")
    for temp_path in [frames_dir, cropped_dir, sam2_masks_dir, output_dir, final_frames_dir]:
        if temp_path.exists():
            shutil.rmtree(temp_path)

    return output_path


def main():
    while True:
        print("="*80)
        print("SAM2 LOCAL - Simple Watermark Removal")
        print("="*80)
        print()
        print("This script will:")
        print("  1. Let you pick a video file")
        print("  2. Use masks from temp_sam2_masks/")
        print("  3. Remove watermarks with ProPainter")
        print("  4. Save result to results/")
        print()

        # Select video
        video_path = select_video()
        if not video_path:
            break

        # Process
        output_path = process_sam2_local(video_path, str(MASKS_FOLDER))

        if output_path:
            print("\n" + "="*80)
            print("SUCCESS!")
            print("="*80)
            print(f"Output: {output_path}")
            print("="*80)

            # Ask to open
            print("\nOpen output video? (y/n): ", end='')
            try:
                answer = input().strip().lower()
                if answer == 'y':
                    os.startfile(output_path)
            except:
                pass
        else:
            print("\n" + "="*80)
            print("FAILED")
            print("="*80)

        # Ask to process another video
        print("\nProcess another video? (y/n): ", end='')
        try:
            answer = input().strip().lower()
            if answer != 'y':
                print("Exiting...")
                break
            print("\n")
        except:
            break


if __name__ == "__main__":
    main()
