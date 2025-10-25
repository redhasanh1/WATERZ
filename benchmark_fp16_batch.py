"""
FP16 TensorRT Batch Benchmark - EXTREME SPEED Without pycuda!
Uses existing FP16 engine with batch processing

Target: 1-2ms per frame (500-1000 fps)
"""

import sys
import cv2
import time
import numpy as np
from pathlib import Path


def extract_frames(video_path, max_frames=300):
    """Extract frames from video into memory"""
    print(f"[*] Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[!] Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Loading {min(max_frames, total_frames)} frames into memory...")

    frames = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()

    print(f"[+] Loaded {len(frames)} frames\n")
    return frames


def benchmark_fp16_batch(frames, batch_size=32):
    """Benchmark FP16 TensorRT engine with batch processing"""
    print("=" * 70)
    print(" " * 15 + "FP16 TensorRT Batch Benchmark")
    print(" " * 20 + "EXTREME SPEED MODE")
    print("=" * 70)
    print()

    # Try batch-enabled engine first, fallback to regular FP16
    engine_paths = [
        "runs/detect/new_sora_watermark/weights/best_fp16_batch_rtx5070.engine",
        "runs/detect/new_sora_watermark/weights/best_fp16.engine"
    ]

    engine_path = None
    for path in engine_paths:
        if Path(path).exists():
            engine_path = path
            break

    if not engine_path:
        print(f"[!] FP16 engine not found!")
        print("   Run BUILD_FP16_BATCH.bat first to create batch-enabled engine")
        return None

    print(f"[*] Loading FP16 TensorRT engine: {engine_path}")

    # Import YOLO
    sys.path.insert(0, 'python_packages')
    from ultralytics import YOLO

    # Load FP16 engine
    model = YOLO(engine_path, task='detect')
    print(f"[+] FP16 Engine loaded!")
    print(f"[*] Batch size: {batch_size}")
    print()

    # Warmup
    print("[*] Warmup run...")
    warmup_batch = frames[:batch_size]
    _ = model(warmup_batch, conf=0.15, device=0, verbose=False)
    print("[+] Warmup complete\n")

    # Benchmark with batch processing
    print(f"[*] Benchmarking {len(frames)} frames with batch size {batch_size}...")
    print(f"[*] Target: 1-2ms per frame (500-1000 fps)")
    print()

    detections_count = 0
    start_time = time.time()

    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]

        # Batch inference
        results = model(batch, conf=0.15, device=0, verbose=False)

        # Count detections
        for result in results:
            detections_count += len(result.boxes)

    end_time = time.time()
    duration = end_time - start_time
    fps = len(frames) / duration
    ms_per_frame = (duration / len(frames)) * 1000

    print(f"[+] Benchmark complete!\n")
    print("=" * 70)
    print(" " * 25 + "RESULTS")
    print("=" * 70)
    print(f"   Total time: {duration:.3f}s")
    print(f"   FPS: {fps:.1f}")
    print(f"   ms per frame: {ms_per_frame:.2f}ms")
    print(f"   Detections: {detections_count}")
    print("=" * 70)
    print()

    # Performance analysis
    if ms_per_frame < 1.0:
        print(f"🔥🔥🔥 EXTREME SPEED! {ms_per_frame:.2f}ms per frame! 🔥🔥🔥")
    elif ms_per_frame < 2.0:
        print(f"🔥🔥 EXCELLENT! {ms_per_frame:.2f}ms per frame!")
    elif ms_per_frame < 5.0:
        print(f"🔥 VERY FAST! {ms_per_frame:.2f}ms per frame!")
    else:
        print(f"⚡ Good: {ms_per_frame:.2f}ms per frame")

    print()
    print("=" * 70)
    print(" " * 15 + "Comparison to Original")
    print("=" * 70)
    print(f"   Original (.pt single frame): 152ms per frame (6.5 fps)")
    print(f"   FP16 TensorRT (single frame): 3.5ms per frame (283 fps)")
    print(f"   FP16 TensorRT (batch {batch_size}): {ms_per_frame:.2f}ms per frame ({fps:.1f} fps)")
    print()
    print(f"   🚀 SPEEDUP vs Original: {152 / ms_per_frame:.1f}x FASTER!")
    print(f"   🚀 SPEEDUP vs FP16 single: {3.5 / ms_per_frame:.1f}x FASTER!")
    print("=" * 70)
    print()

    return {
        'duration': duration,
        'fps': fps,
        'ms_per_frame': ms_per_frame,
        'detections': detections_count
    }


def main():
    print()

    # Get video path
    if len(sys.argv) < 2:
        # Use default test video
        default_videos = [
            "uploads/05d90319-fef7-41fc-8103-449ccae86845.mp4",
            "temp/video_cache/05d90319-fef7-41fc-8103-449ccae86845.mp4"
        ]

        video_path = None
        for vid in default_videos:
            if Path(vid).exists():
                video_path = vid
                break

        if not video_path:
            print("Usage: python benchmark_fp16_batch.py <path_to_video>")
            print()
            print("Example:")
            print("  python benchmark_fp16_batch.py uploads/test_video.mp4")
            sys.exit(1)

        print(f"[*] Using default video: {video_path}\n")
    else:
        video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"[!] Video not found: {video_path}")
        sys.exit(1)

    # Extract frames
    frames = extract_frames(video_path, max_frames=300)

    if len(frames) == 0:
        print("[!] No frames extracted!")
        sys.exit(1)

    # Test different batch sizes
    batch_sizes = [32, 64]

    results = {}
    for batch_size in batch_sizes:
        result = benchmark_fp16_batch(frames, batch_size=batch_size)
        if result:
            results[batch_size] = result
            print()
            input("[*] Press Enter to continue to next batch size...")
            print("\n" * 2)

    # Summary
    if len(results) > 1:
        print("=" * 70)
        print(" " * 20 + "BATCH SIZE COMPARISON")
        print("=" * 70)
        print(f"{'Batch Size':<15} {'FPS':<15} {'ms/frame':<15} {'Speedup':<15}")
        print("-" * 70)

        baseline_ms = 152  # Original speed
        for batch_size, result in results.items():
            speedup = baseline_ms / result['ms_per_frame']
            print(f"{batch_size:<15} {result['fps']:<15.1f} {result['ms_per_frame']:<15.2f} {speedup:<15.1f}x")

        print("=" * 70)
        print()

    print("[+] Benchmark complete!")
    print()


if __name__ == "__main__":
    main()
