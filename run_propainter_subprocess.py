#!/usr/bin/env python3
"""
Subprocess wrapper for ProPainter execution.

This script runs ProPainter in a separate process to avoid blocking the
Celery worker. The worker can spawn this subprocess and immediately return
to polling the Redis queue for more tasks.

Usage:
    python run_propainter_subprocess.py \
        --frames_dir /path/to/frames \
        --masks_dir /path/to/masks \
        --output_dir /path/to/output \
        --flow_backend raft \
        --fp16
"""

import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run ProPainter in subprocess')
    parser.add_argument('--frames_dir', required=True, help='Directory containing input frames')
    parser.add_argument('--masks_dir', required=True, help='Directory containing mask frames')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--flow_backend', default='raft', choices=['raft', 'fastflownet'],
                        help='Flow backend to use')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 for inference')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Neighbor length for temporal propagation')
    parser.add_argument('--ref_stride', type=int, default=10, help='Reference frame stride')
    parser.add_argument('--raft_iter', type=int, default=20, help='Number of RAFT iterations')

    args = parser.parse_args()

    # Get script directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Add faster-propainter to path
    faster_propainter_path = os.path.join(SCRIPT_DIR, 'faster-propainter-main')
    if faster_propainter_path not in sys.path:
        sys.path.insert(0, faster_propainter_path)

    # Import in subprocess (fresh CUDA context)
    from watermark import pipeline as faster_propainter_pipeline

    print(f"[ProPainter Subprocess] Starting...")
    print(f"  Frames: {args.frames_dir}")
    print(f"  Masks: {args.masks_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Flow backend: {args.flow_backend}")
    print(f"  FP16: {args.fp16}")

    try:
        # Run ProPainter - pipeline is the function itself
        faster_propainter_pipeline(
            video=args.frames_dir,
            mask=args.masks_dir,
            output=args.output_dir,
            resize_ratio=1.0,
            mask_dilation=4,
            ref_stride=args.ref_stride,
            neighbor_length=args.neighbor_length,
            subvideo_length=80,
            raft_iter=args.raft_iter,
            mode="video_inpainting",
            save_frames=True,
            fp16=args.fp16
        )

        print(f"[ProPainter Subprocess] ✅ Complete!")
        sys.exit(0)

    except Exception as e:
        print(f"[ProPainter Subprocess] ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
