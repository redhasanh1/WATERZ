"""
Process a single video chunk in an isolated subprocess.

When this process exits, the OS forcibly reclaims ALL GPU memory.
This is the only reliable way to free TensorRT engines and PyTorch allocator pools.

Usage:
    python process_chunk_subprocess.py '{"video_path": "...", "start_frame": 0, ...}'
"""
import sys
import json
import os
import traceback

# Set environment before any CUDA imports
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')


def main():
    if len(sys.argv) < 2:
        print("[ERROR] No chunk args provided", file=sys.stderr)
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])

        # All heavy imports INSIDE main() to ensure fresh CUDA context
        import cv2
        import numpy as np
        import torch

        # Add ProPainter to path
        sys.path.insert(0, r'D:\watermarkz\faster-propainter-main')
        from watermark import pipeline as faster_propainter_pipeline

        # Extract args
        video_path = args['video_path']
        start_frame = args['start_frame']
        end_frame = args['end_frame']
        masks_dir = args['masks_dir']
        chunk_output_dir = args['chunk_output_dir']
        final_frames_dir = args['final_frames_dir']
        crop_x = args['crop_x']
        crop_y = args['crop_y']
        crop_w = args['crop_w']
        crop_h = args['crop_h']
        width = args['width']
        height = args['height']
        mask_dilation = args.get('mask_dilation', 4)
        original_fps = args.get('original_fps', 30)
        chunk_idx = args.get('chunk_idx', 0)
        total_chunks = args.get('total_chunks', 1)

        print(f"[SUBPROCESS] Processing chunk {chunk_idx+1}/{total_chunks} (frames {start_frame}-{end_frame-1})", flush=True)

        # --- 1. Load frames for this chunk ---
        print(f"[SUBPROCESS] Loading frames {start_frame}-{end_frame-1}...", flush=True)
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            # Crop to region of interest
            cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            frames.append(np.ascontiguousarray(cropped))
        cap.release()
        print(f"[SUBPROCESS] Loaded {len(frames)} frames", flush=True)

        # --- 2. Load masks for this chunk ---
        print(f"[SUBPROCESS] Loading masks {start_frame}-{end_frame-1}...", flush=True)
        masks = []
        for frame_idx in range(start_frame, end_frame):
            mask_file = os.path.join(masks_dir, f'{frame_idx:05d}.png')
            if os.path.exists(mask_file):
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Crop mask to same region
                    mask_cropped = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                    masks.append(np.ascontiguousarray(mask_cropped))
                else:
                    masks.append(np.zeros((crop_h, crop_w), dtype=np.uint8))
            else:
                masks.append(np.zeros((crop_h, crop_w), dtype=np.uint8))
        print(f"[SUBPROCESS] Loaded {len(masks)} masks", flush=True)

        # --- 3. Run ProPainter ---
        print(f"[SUBPROCESS] Running ProPainter...", flush=True)
        os.makedirs(chunk_output_dir, exist_ok=True)

        # Force FP32 - fresh models don't have proper FP16 conversion like cached ones
        # (FP16 causes "Input type (Half) and bias type (float) should be the same" error)
        faster_propainter_pipeline(
            video='dummy', mask='dummy', output=chunk_output_dir,
            resize_ratio=1.0, mask_dilation=mask_dilation,
            ref_stride=15, neighbor_length=3, subvideo_length=60,
            raft_iter=10, mode="video_inpainting", save_fps=int(original_fps),
            save_frames=True, fp16=False, use_cached_models=False,  # FP32 + no cache (subprocess exits anyway)
            frames_array=frames, masks_array=masks
        )

        # Free input arrays
        del frames
        del masks

        # --- 4. Find output frames ---
        frames_subdir = None
        for d in os.listdir(chunk_output_dir):
            candidate = os.path.join(chunk_output_dir, d, 'frames')
            if os.path.isdir(candidate):
                frames_subdir = candidate
                break

        if not frames_subdir:
            print(f"[ERROR] No output frames found in {chunk_output_dir}", file=sys.stderr, flush=True)
            sys.exit(1)

        # --- 5. Stream composite directly to final_frames_dir (WRITE-THROUGH) ---
        print(f"[SUBPROCESS] Streaming composite to {final_frames_dir}...", flush=True)
        os.makedirs(final_frames_dir, exist_ok=True)
        output_files = sorted([f for f in os.listdir(frames_subdir) if f.endswith('.png')])

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        total_output = len(output_files)

        for i, out_file in enumerate(output_files):
            frame_idx = start_frame + i
            ret, orig_frame = cap.read()
            if not ret:
                break
            out_frame = cv2.imread(os.path.join(frames_subdir, out_file))
            if out_frame is not None:
                orig_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = out_frame

            # WRITE-THROUGH: Write immediately to disk, don't buffer
            output_path = os.path.join(final_frames_dir, f'{frame_idx:05d}.png')
            cv2.imwrite(output_path, orig_frame)
            frames_written += 1

            # Progress indication every 30 frames
            if frames_written % 30 == 0 or frames_written == total_output:
                print(f"[SUBPROCESS] Written {frames_written}/{total_output} frames...", flush=True)

        cap.release()

        # Log VRAM before cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[SUBPROCESS] VRAM before exit: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved", flush=True)

        print(f"[SUBPROCESS] Chunk {chunk_idx+1} complete: wrote {frames_written} frames to {final_frames_dir}", flush=True)
        print(f"[SUBPROCESS] Exiting - OS will reclaim ALL GPU memory", flush=True)

        # Exit cleanly - OS will forcibly free all GPU memory
        sys.exit(0)

    except Exception as e:
        print(f"[SUBPROCESS ERROR] {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)


if __name__ == '__main__':
    main()
