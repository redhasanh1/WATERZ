"""
NVOF Real Video Test

Test NVOF wrapper on a real 300-frame video from videostotrain folder.
Compare performance against RAFT TensorRT baseline.
"""

import torch
import time
import cv2
from pathlib import Path
import numpy as np
from nvof_wrapper import NVOFOpticalFlow


def load_video_frames(video_path: Path, max_frames: int = 300) -> torch.Tensor:
    """
    Load video frames as PyTorch tensor.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load

    Returns:
        Tensor of shape (1, T, 3, H, W) in range [0, 1]
    """
    print(f"Loading video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")

    frames = []
    frame_count = 0

    while frame_count < min(max_frames, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 480x480 if needed (NVOF expects fixed size)
        if height != 480 or width != 480:
            frame_rgb = cv2.resize(frame_rgb, (480, 480), interpolation=cv2.INTER_LINEAR)

        # Convert to float32 [0, 1] and CHW format
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW

        frames.append(frame_tensor)
        frame_count += 1

    cap.release()

    # Stack frames into (T, C, H, W) and add batch dimension
    frames_tensor = torch.stack(frames, dim=0).unsqueeze(0)  # (1, T, C, H, W)

    print(f"  Loaded: {frame_count} frames")
    print(f"  Tensor shape: {frames_tensor.shape}")

    return frames_tensor


def find_video_file(folder: Path) -> Path:
    """Find first video file in folder."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    for ext in video_extensions:
        videos = list(folder.glob(f'*{ext}'))
        if videos:
            return videos[0]

    raise FileNotFoundError(f"No video files found in: {folder}")


def test_nvof_real_video():
    """Test NVOF on real 300-frame video."""
    print("=" * 70)
    print(" NVOF Real Video Test - 300 Frames")
    print(" RTX 5070 Blackwell GPU")
    print("=" * 70)
    print()

    # Find video file
    videos_folder = Path("E:/watermarkz/videostotrain")
    if not videos_folder.exists():
        print(f"[ERROR] Folder not found: {videos_folder}")
        print("        Please provide correct path to videostotrain folder")
        return

    try:
        video_path = find_video_file(videos_folder)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    print(f"[1/4] Found video: {video_path.name}")
    print()

    # Load video frames
    print("[2/4] Loading video frames...")
    try:
        frames = load_video_frames(video_path, max_frames=300)
        frames = frames.cuda()  # Move to GPU
    except Exception as e:
        print(f"[ERROR] Failed to load video: {e}")
        import traceback
        traceback.print_exc()
        return

    B, T, C, H, W = frames.shape
    num_pairs = T - 1

    print()
    print(f"Video loaded:")
    print(f"  Batch: {B}")
    print(f"  Frames: {T}")
    print(f"  Channels: {C}")
    print(f"  Resolution: {H}x{W}")
    print(f"  Optical flow pairs: {num_pairs}")
    print()

    # Initialize NVOF
    print("[3/4] Initializing NVOF...")
    try:
        nvof = NVOFOpticalFlow(width=W, height=H, device_id=0)
        print("[OK] NVOF initialized")
    except Exception as e:
        print(f"[ERROR] NVOF initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # Warmup
    print("[4/4] Processing video with NVOF...")
    print("  Warmup: Processing first 10 frames...")
    warmup_frames = frames[:, :11, :, :, :]  # 11 frames = 10 pairs
    try:
        _ = nvof(warmup_frames)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[ERROR] Warmup failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("  [OK] Warmup complete")
    print()

    # Full video processing
    print(f"  Processing full video ({T} frames, {num_pairs} pairs)...")
    print("  This will take a few seconds...")
    print()

    try:
        torch.cuda.synchronize()
        start = time.perf_counter()

        flows_fwd, flows_bwd = nvof(frames)

        torch.cuda.synchronize()
        end = time.perf_counter()

        total_time_ms = (end - start) * 1000
        avg_time_ms = total_time_ms / num_pairs

        print("[OK] Processing complete!")
        print()

        # Results
        print("=" * 70)
        print(" RESULTS")
        print("=" * 70)
        print()

        print(f"NVOF Hardware Optical Flow:")
        print(f"  Total frames: {T}")
        print(f"  Optical flow pairs: {num_pairs}")
        print(f"  Total processing time: {total_time_ms:.2f} ms ({total_time_ms / 1000:.3f} seconds)")
        print(f"  Average per pair: {avg_time_ms:.3f} ms")
        print(f"  Throughput: {num_pairs / (total_time_ms / 1000):.1f} pairs/sec")
        print()

        print(f"Flow Statistics:")
        print(f"  Forward flow shape: {flows_fwd.shape}")
        print(f"  Backward flow shape: {flows_bwd.shape if flows_bwd is not None else None}")
        print(f"  Forward flow range: [{flows_fwd.min().item():.2f}, {flows_fwd.max().item():.2f}] pixels")
        if flows_bwd is not None:
            print(f"  Backward flow range: [{flows_bwd.min().item():.2f}, {flows_bwd.max().item():.2f}] pixels")
        print()

        # Comparison with RAFT TensorRT
        raft_trt_avg = 3.64  # ms per pair
        raft_total_ms = raft_trt_avg * num_pairs

        print("=" * 70)
        print(" COMPARISON vs RAFT TensorRT")
        print("=" * 70)
        print()

        print(f"RAFT TensorRT (baseline):")
        print(f"  Estimated total time: {raft_total_ms:.2f} ms ({raft_total_ms / 1000:.3f} seconds)")
        print(f"  Average per pair: {raft_trt_avg:.3f} ms")
        print(f"  Throughput: {num_pairs / (raft_total_ms / 1000):.1f} pairs/sec")
        print()

        speedup = raft_total_ms / total_time_ms
        time_saved_ms = raft_total_ms - total_time_ms
        time_saved_sec = time_saved_ms / 1000

        print(f"Speedup Analysis:")
        print(f"  Real-world speedup: {speedup:.2f}x")
        print(f"  Time saved: {time_saved_ms:.2f} ms ({time_saved_sec:.3f} seconds)")
        print()

        if speedup > 1.5:
            print(f"[OK] NVOF is {speedup:.2f}x faster - worth integrating!")
        elif speedup > 1.0:
            print(f"[MARGINAL] NVOF is only {speedup:.2f}x faster - small gain")
            print(f"           Data conversion overhead is significant")
        else:
            print(f"[FAIL] NVOF is slower than RAFT TensorRT!")

        print()
        print("=" * 70)

        # Per-frame time analysis
        print()
        print("Per-frame breakdown:")
        print(f"  Expected (C++ benchmark): ~1.30 ms/pair")
        print(f"  Actual (Python wrapper):  {avg_time_ms:.3f} ms/pair")
        print(f"  Overhead:                 {avg_time_ms - 1.30:.3f} ms/pair")
        print(f"  Overhead percentage:      {((avg_time_ms - 1.30) / 1.30 * 100):.1f}%")
        print()

        components_ms = {
            "NVOF hardware execution": 1.30,
            "RGB→Grayscale conversion": (avg_time_ms - 1.30) * 0.3,
            "Float→uint8 conversion": (avg_time_ms - 1.30) * 0.2,
            "CUDA memcpy (upload)": (avg_time_ms - 1.30) * 0.2,
            "CUDA memcpy (download)": (avg_time_ms - 1.30) * 0.2,
            "S10.5→Float conversion": (avg_time_ms - 1.30) * 0.1,
        }

        print("Estimated time breakdown:")
        for component, time_ms in components_ms.items():
            percentage = (time_ms / avg_time_ms) * 100
            print(f"  {component:30s}: {time_ms:6.3f} ms ({percentage:5.1f}%)")

        print()
        print("=" * 70)

        return True

    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_nvof_real_video()
