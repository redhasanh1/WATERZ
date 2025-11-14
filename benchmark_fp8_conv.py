"""
Benchmark FP8 Conv2d/Conv3d optimizations for Encoder/Decoder and RFC Net
Tests with FP8 ON vs OFF and measures performance gains
"""
import os
import sys
import time
import torch
import numpy as np

# Fix Unicode encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Setup paths
sys.path.insert(0, 'python_packages')
sys.path.insert(0, 'faster-propainter-main')
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')

print("="*80)
print("FP8 CONV2D/CONV3D BENCHMARK - Encoder/Decoder + RFC Net")
print("="*80)
print()

def run_test(enable_fp8_encoder, enable_fp8_decoder, enable_fp8_rfcnet, test_name):
    """Run ProPainter with specific FP8 settings"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")
    print(f"  ENABLE_FP8_ENCODER={enable_fp8_encoder}")
    print(f"  ENABLE_FP8_DECODER={enable_fp8_decoder}")
    print(f"  ENABLE_FP8_RFCNET={enable_fp8_rfcnet}")
    print()

    # Set environment variables
    os.environ['ENABLE_FP8_ENCODER'] = '1' if enable_fp8_encoder else '0'
    os.environ['ENABLE_FP8_DECODER'] = '1' if enable_fp8_decoder else '0'
    os.environ['ENABLE_FP8_RFCNET'] = '1' if enable_fp8_rfcnet else '0'

    # Always enable other optimizations for fair comparison
    os.environ['USE_NEUFLOW'] = '1'
    os.environ['ENABLE_FLASH_ATTENTION'] = '1'
    os.environ['ENABLE_FP8_TRANSFORMER'] = '1'

    # Clear module cache to force reload with new settings
    if 'model.propainter' in sys.modules:
        del sys.modules['model.propainter']
    if 'model.recurrent_flow_completion' in sys.modules:
        del sys.modules['model.recurrent_flow_completion']
    if 'watermark' in sys.modules:
        del sys.modules['watermark']

    # Import fresh modules
    from watermark import pipeline as propainter_pipeline
    import cv2

    # Test video setup
    test_video = 'videostotrain/first.mp4'
    output_dir = f'results/benchmark_fp8_{test_name.replace(" ", "_").lower()}'

    if not os.path.exists(test_video):
        print(f"[ERROR] Test video not found: {test_video}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Load 30 frames for benchmark
    cap = cv2.VideoCapture(test_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    frame_limit = 30
    for i in range(frame_limit):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"  Frames: {len(frames)}, Resolution: {width}x{height}")

    # Create dummy masks (20% masked area)
    masks = []
    for frame in frames:
        mask = np.zeros((height, width), dtype=np.uint8)
        h_start, h_end = height//3, 2*height//3
        w_start, w_end = width//3, 2*width//3
        mask[h_start:h_end, w_start:w_end] = 255
        masks.append(mask)

    frames_array = [np.ascontiguousarray(f) for f in frames]
    masks_array = [np.ascontiguousarray(m) for m in masks]

    # Warmup run (load models, compile kernels)
    print("  Running warmup pass...")
    try:
        propainter_pipeline(
            video=test_video,
            mask='dummy',
            output=output_dir,
            resize_ratio=1.0,
            mask_dilation=4,
            ref_stride=15,
            neighbor_length=10,
            subvideo_length=120,
            raft_iter=10,
            mode="video_inpainting",
            save_fps=fps,
            save_frames=False,  # Don't save during warmup
            fp16=True,
            use_cached_models=True,
            frames_array=frames_array[:10],  # Only 10 frames for warmup
            masks_array=masks_array[:10]
        )
    except Exception as e:
        print(f"  [WARNING] Warmup failed: {e}")

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Actual benchmark run
    print("  Running benchmark pass...")
    start_time = time.time()

    try:
        propainter_pipeline(
            video=test_video,
            mask='dummy',
            output=output_dir,
            resize_ratio=1.0,
            mask_dilation=4,
            ref_stride=15,
            neighbor_length=10,
            subvideo_length=120,
            raft_iter=10,
            mode="video_inpainting",
            save_fps=fps,
            save_frames=True,
            fp16=True,
            use_cached_models=True,
            frames_array=frames_array,
            masks_array=masks_array
        )
    except Exception as e:
        print(f"  [ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    per_frame_time = (total_time / len(frames)) * 1000  # ms

    print(f"\n  RESULTS:")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Per-frame: {per_frame_time:.1f}ms")
    print(f"    FPS: {len(frames)/total_time:.1f}")

    return {
        'name': test_name,
        'total_time': total_time,
        'per_frame_ms': per_frame_time,
        'fps': len(frames)/total_time,
        'frames': len(frames)
    }

# Run benchmarks
print("\nStarting benchmarks (this will take several minutes)...\n")

results = []

# Test 1: FP8 Disabled (Baseline)
result = run_test(
    enable_fp8_encoder=False,
    enable_fp8_decoder=False,
    enable_fp8_rfcnet=False,
    test_name="FP8 OFF (Baseline)"
)
if result:
    results.append(result)

# Test 2: FP8 Enabled (All)
result = run_test(
    enable_fp8_encoder=True,
    enable_fp8_decoder=True,
    enable_fp8_rfcnet=True,
    test_name="FP8 ON (Encoder+Decoder+RFCNet)"
)
if result:
    results.append(result)

# Test 3: FP8 Encoder+Decoder Only
result = run_test(
    enable_fp8_encoder=True,
    enable_fp8_decoder=True,
    enable_fp8_rfcnet=False,
    test_name="FP8 Encoder+Decoder Only"
)
if result:
    results.append(result)

# Test 4: FP8 RFCNet Only
result = run_test(
    enable_fp8_encoder=False,
    enable_fp8_decoder=False,
    enable_fp8_rfcnet=True,
    test_name="FP8 RFCNet Only"
)
if result:
    results.append(result)

# Print comparison
print("\n" + "="*80)
print("BENCHMARK COMPARISON")
print("="*80)
print()

if len(results) < 2:
    print("[ERROR] Not enough successful runs to compare")
    exit(1)

baseline = results[0]
print(f"{'Configuration':<40} {'Time (s)':<12} {'Per-Frame (ms)':<18} {'Speedup':<10}")
print("-"*80)

for result in results:
    speedup = baseline['total_time'] / result['total_time']
    speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"
    print(f"{result['name']:<40} {result['total_time']:<12.2f} {result['per_frame_ms']:<18.1f} {speedup_str:<10}")

print()
print("="*80)
print("ANALYSIS")
print("="*80)
print()

if len(results) >= 2:
    full_fp8 = results[1]
    speedup = baseline['total_time'] / full_fp8['total_time']
    time_saved = baseline['total_time'] - full_fp8['total_time']
    pct_faster = (speedup - 1) * 100

    print(f"Overall FP8 Optimization:")
    print(f"  Baseline: {baseline['total_time']:.2f}s")
    print(f"  With FP8: {full_fp8['total_time']:.2f}s")
    print(f"  Time saved: {time_saved:.2f}s ({pct_faster:.1f}% faster)")
    print(f"  Speedup: {speedup:.2f}x")
    print()

    if len(results) >= 4:
        enc_dec = results[2]
        rfcnet = results[3]

        enc_dec_contribution = baseline['total_time'] - enc_dec['total_time']
        rfcnet_contribution = baseline['total_time'] - rfcnet['total_time']

        print(f"Individual Component Contributions:")
        print(f"  Encoder+Decoder FP8: {enc_dec_contribution:.2f}s saved ({(enc_dec_contribution/baseline['total_time']*100):.1f}% improvement)")
        print(f"  RFCNet FP8: {rfcnet_contribution:.2f}s saved ({(rfcnet_contribution/baseline['total_time']*100):.1f}% improvement)")
        print()

print("="*80)
print("BENCHMARK COMPLETE")
print("="*80)
