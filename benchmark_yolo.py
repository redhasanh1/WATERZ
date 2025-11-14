"""
Simple YOLO Benchmark: Compare PyTorch .pt vs TensorRT .engine
Tests on 300 frames from a video to show REAL speedup
"""
import sys
import cv2
import time
import numpy as np
from pathlib import Path

def extract_frames(video_path, max_frames=300):
    """Extract frames from video into memory"""
    print(f"📹 Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
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

    print(f"✅ Loaded {len(frames)} frames\n")
    return frames


def benchmark_pytorch(frames):
    """Benchmark with PyTorch .pt model"""
    print("━" * 50)
    print("TEST 1: PyTorch Model (.pt)")
    print("━" * 50)

    pt_path = "runs/detect/new_sora_watermark/weights/best.pt"

    if not Path(pt_path).exists():
        print(f"❌ PyTorch model not found: {pt_path}")
        return None

    print(f"Loading: {pt_path}")

    # Import YOLO
    sys.path.insert(0, 'python_packages')
    from ultralytics import YOLO

    # Load PyTorch model
    model = YOLO(pt_path)
    print("✅ Model loaded\n")

    # Warmup (first inference is always slower)
    print("🔥 Warmup run...")
    _ = model(frames[0], conf=0.25, device=0, verbose=False)
    print("✅ Warmup complete\n")

    # Benchmark
    print(f"🚀 Processing {len(frames)} frames...")
    detections = 0

    start_time = time.time()

    for frame in frames:
        results = model(frame, conf=0.25, device=0, verbose=False)
        # Count detections
        for result in results:
            detections += len(result.boxes)

    end_time = time.time()
    duration = end_time - start_time
    fps = len(frames) / duration

    print(f"✅ Complete!\n")
    print(f"   Duration: {duration:.2f}s")
    print(f"   FPS: {fps:.2f}")
    print(f"   Detections: {detections}")
    print()

    return {
        'duration': duration,
        'fps': fps,
        'detections': detections
    }


def benchmark_tensorrt(frames):
    """Benchmark with TensorRT .engine model"""
    print("━" * 50)
    print("TEST 2: TensorRT Engine (.engine)")
    print("━" * 50)

    engine_path = "runs/detect/new_sora_watermark/weights/best_fp16.engine"

    if not Path(engine_path).exists():
        print(f"❌ TensorRT engine not found: {engine_path}")
        print("   Run build_yolo_trt_engine.py first!")
        return None

    print(f"Loading: {engine_path}")

    # Import YOLO
    sys.path.insert(0, 'python_packages')
    from ultralytics import YOLO

    # Load TensorRT engine
    model = YOLO(engine_path, task='detect')
    print("✅ Engine loaded\n")

    # Warmup
    print("🔥 Warmup run...")
    _ = model(frames[0], conf=0.25, device=0, verbose=False)
    print("✅ Warmup complete\n")

    # Benchmark
    print(f"🚀 Processing {len(frames)} frames...")
    detections = 0

    start_time = time.time()

    for frame in frames:
        results = model(frame, conf=0.25, device=0, verbose=False)
        # Count detections
        for result in results:
            detections += len(result.boxes)

    end_time = time.time()
    duration = end_time - start_time
    fps = len(frames) / duration

    print(f"✅ Complete!\n")
    print(f"   Duration: {duration:.2f}s")
    print(f"   FPS: {fps:.2f}")
    print(f"   Detections: {detections}")
    print()

    return {
        'duration': duration,
        'fps': fps,
        'detections': detections
    }


def show_comparison(pt_result, trt_result):
    """Show side-by-side comparison"""
    print("\n")
    print("=" * 70)
    print(" " * 20 + "🔬 BENCHMARK RESULTS")
    print("=" * 70)
    print()

    if pt_result and trt_result:
        speedup = pt_result['duration'] / trt_result['duration']
        fps_gain = trt_result['fps'] / pt_result['fps']

        print(f"{'Model':<30} {'Duration':<15} {'FPS':<15} {'Detections':<15}")
        print("-" * 70)
        print(f"{'PyTorch (.pt)':<30} {pt_result['duration']:>10.2f}s    {pt_result['fps']:>10.2f}    {pt_result['detections']:>10}")
        print(f"{'TensorRT (.engine)':<30} {trt_result['duration']:>10.2f}s    {trt_result['fps']:>10.2f}    {trt_result['detections']:>10}")
        print("-" * 70)
        print()
        print(f"🚀 SPEEDUP: {speedup:.1f}x faster!")
        print(f"📈 FPS GAIN: {fps_gain:.1f}x more frames per second!")
        print()

        if trt_result['detections'] == pt_result['detections']:
            print("✅ Detection counts match (accuracy preserved)")
        else:
            diff = abs(trt_result['detections'] - pt_result['detections'])
            print(f"⚠️  Detection count difference: {diff} (FP16 precision differences)")

    elif pt_result:
        print("PyTorch Results:")
        print(f"   Duration: {pt_result['duration']:.2f}s")
        print(f"   FPS: {pt_result['fps']:.2f}")
        print(f"   Detections: {pt_result['detections']}")
        print("\n❌ TensorRT test failed - see errors above")

    elif trt_result:
        print("TensorRT Results:")
        print(f"   Duration: {trt_result['duration']:.2f}s")
        print(f"   FPS: {trt_result['fps']:.2f}")
        print(f"   Detections: {trt_result['detections']}")
        print("\n❌ PyTorch test failed - see errors above")

    print("=" * 70)
    print()


def main():
    print()
    print("=" * 70)
    print(" " * 15 + "🔬 YOLO BENCHMARK: .pt vs .engine")
    print("=" * 70)
    print()

    # Get video path
    if len(sys.argv) < 2:
        print("Usage: python benchmark_yolo.py <path_to_video>")
        print()
        print("Example:")
        print("  python benchmark_yolo.py uploads/test_video.mp4")
        print("  python benchmark_yolo.py temp/video_cache/05d90319-fef7-41fc-8103-449ccae86845.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    # Extract frames
    frames = extract_frames(video_path, max_frames=300)

    if len(frames) == 0:
        print("❌ No frames extracted!")
        sys.exit(1)

    # Run benchmarks
    pt_result = benchmark_pytorch(frames)
    trt_result = benchmark_tensorrt(frames)

    # Show comparison
    show_comparison(pt_result, trt_result)


if __name__ == "__main__":
    main()
