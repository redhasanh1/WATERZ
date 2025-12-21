#!/usr/bin/env python3
"""
SAM2 Tracking Script for WSL2 - Called from Windows Celery worker
Usage:
  Point mode:  python sam2_track_wsl2.py <video_path> --point x,y <output_masks_dir>
  Bbox mode:   python sam2_track_wsl2.py <video_path> --bbox x1,y1,x2,y2 <output_masks_dir>

Uses HYBRID approach:
- Frame 0: TensorRT FP16 (instant)
- Frames 1-N: PyTorch torch.compile() + BFloat16
"""

import os
import sys
import cv2
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
import argparse
import subprocess

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
SAM2_PATH = Path(__file__).parent / "segment-anything-2"
sys.path.insert(0, str(SAM2_PATH))

# Import SAM2 PyTorch
from sam2.build_sam import build_sam2_video_predictor

# Try to import TensorRT predictor (optional for frame 0)
try:
    from sam2_trt_predictor import SAM2TensorRTPredictor
    HAS_TRT = True
except ImportError:
    HAS_TRT = False
    print("[SAM2-WSL2] TensorRT predictor not available, using PyTorch for all frames")

# Try to import DALI for GPU-direct video decoding
try:
    from nvidia.dali import pipeline_def, fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    HAS_DALI = True
    print("[SAM2-WSL2] NVIDIA DALI available - GPU-direct video streaming enabled!")
except ImportError:
    HAS_DALI = False
    print("[SAM2-WSL2] DALI not available, using CPU frame loading")


# =============================================================================
# REVOLUTIONARY COMPRESSION CLASSES
# =============================================================================

def extract_to_h265(video_path, output_path, target_fps=None, max_height=720):
    """
    Extract video to H.265 for GPU-accelerated decoding.
    Way smaller than JPG frames: ~70MB vs 4GB+ for 2000 frames.
    """
    import shlex

    # Get source video info
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate resize
    if orig_height > max_height:
        scale = max_height / orig_height
        new_width = int(orig_width * scale) // 2 * 2  # Ensure even
        new_height = max_height // 2 * 2
    else:
        new_width, new_height = orig_width, orig_height

    fps_filter = f",fps={target_fps}" if target_fps else ""

    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-c:v', 'libx265', '-crf', '18', '-preset', 'fast',
        '-vf', f'scale={new_width}:{new_height}{fps_filter}',
        '-pix_fmt', 'yuv420p',
        '-an',  # No audio
        output_path
    ]

    print(f"[H.265] Converting {orig_width}x{orig_height} -> {new_width}x{new_height}")
    subprocess.run(cmd, check=True, capture_output=True)

    size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"[H.265] Created {size_mb:.1f}MB video (vs ~{total_frames * 0.2:.0f}MB in JPGs)")

    return total_frames, orig_fps, orig_width, orig_height, new_width, new_height


class DALIFrameStreamer:
    """
    GPU-direct video streaming using NVIDIA DALI with RANDOM ACCESS.
    Decodes H.265 directly on GPU - ZERO CPU RAM usage!

    Supports __getitem__ for random frame access needed by SAM2.
    """

    def __init__(self, video_path, device_id=0, image_size=1024, max_height=720):
        if not HAS_DALI:
            raise RuntimeError("DALI not installed! pip install nvidia-dali-cuda120")

        self.device_id = device_id
        self.image_size = image_size
        self.max_height = max_height

        # Convert to H.265 for optimal DALI decoding
        self.h265_path = self._convert_to_h265(video_path)
        self.video_path = self.h265_path

        # SAM2's normalization constants
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(device_id)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(device_id)

        # Build pipeline
        self._build_pipeline()

        # Cache for frames (optional - for repeated access)
        self._cache = {}
        self._cache_enabled = False

    def _convert_to_h265(self, input_path):
        """Convert video to DALI-friendly H.265 stream"""
        import tempfile

        # Get video info first
        cap = cv2.VideoCapture(input_path)
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        # Calculate resize
        if orig_height > self.max_height:
            scale = self.max_height / orig_height
            new_width = int(orig_width * scale) // 2 * 2
            new_height = self.max_height // 2 * 2
        else:
            new_width, new_height = orig_width, orig_height

        self.video_height = new_height
        self.video_width = new_width

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()

        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx265', '-crf', '18', '-preset', 'ultrafast',
            '-vf', f'scale={new_width}:{new_height}',
            '-pix_fmt', 'yuv420p',
            '-an',
            temp_path
        ]

        print(f"[DALI] Converting to H.265: {orig_width}x{orig_height} -> {new_width}x{new_height}")
        subprocess.run(cmd, capture_output=True, check=True)

        size_mb = os.path.getsize(temp_path) / (1024**2)
        print(f"[DALI] H.265 video: {size_mb:.1f}MB")

        return temp_path

    def _build_pipeline(self):
        @pipeline_def
        def video_pipe():
            video = fn.readers.video(
                device="gpu",
                filenames=[self.video_path],
                sequence_length=1,
                shard_id=0,
                num_shards=1,
                random_shuffle=False,
                initial_fill=2,
                image_type=types.RGB,
                dtype=types.FLOAT,
                normalized=False,
                name="video_reader"
            )
            # Resize to SAM2's internal size
            video = fn.resize(video, size=[self.image_size, self.image_size])
            # Normalize to 0-1
            video = video / 255.0
            return video

        self.pipe = video_pipe(
            batch_size=1,
            num_threads=2,
            device_id=self.device_id,
            prefetch_queue_depth=1  # Minimal VRAM
        )
        self.pipe.build()

        # Get total frames
        self._length = self.pipe.reader_meta("video_reader")["epoch_size"]
        self._current_idx = 0
        print(f"[DALI] Ready: {self._length} frames, VRAM per frame: ~{self.image_size*self.image_size*3*4/1024/1024:.1f}MB")

    def __len__(self):
        return self._length

    def __getitem__(self, frame_idx):
        """Random access to frame - used by SAM2 during propagation"""
        if frame_idx < 0 or frame_idx >= self._length:
            raise IndexError(f"Frame {frame_idx} out of bounds (0-{self._length-1})")

        # Check cache
        if self._cache_enabled and frame_idx in self._cache:
            return self._cache[frame_idx]

        # Seek to frame by resetting and skipping
        # This is necessary because DALI video reader doesn't support random seek directly
        if frame_idx < self._current_idx:
            self.pipe.reset()
            self._current_idx = 0

        # Skip to target frame
        while self._current_idx < frame_idx:
            self.pipe.run()
            self._current_idx += 1

        # Get the target frame
        output = self.pipe.run()
        self._current_idx += 1

        frame = output[0].as_tensor()

        # Convert to PyTorch tensor [B, T, H, W, C] -> [C, H, W]
        frame_torch = torch.from_dlpack(frame).squeeze(0).squeeze(0).permute(2, 0, 1)

        # Apply SAM2 normalization
        frame_torch = frame_torch.unsqueeze(0)  # Add batch dim for normalization
        frame_torch = (frame_torch - self.img_mean) / self.img_std
        frame_torch = frame_torch.squeeze(0)  # Remove batch dim

        # Cache if enabled
        if self._cache_enabled:
            self._cache[frame_idx] = frame_torch

        return frame_torch  # [C, H, W]

    def __iter__(self):
        """Sequential iteration (more efficient than random access)"""
        self.pipe.reset()
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self._length:
            raise StopIteration
        return self[self._current_idx]

    def preload_all(self):
        """Preload all frames to GPU (uses VRAM but fast access)"""
        print(f"[DALI] Preloading {self._length} frames to VRAM...")
        self._cache_enabled = True
        self.pipe.reset()
        self._current_idx = 0

        frames = []
        for i in tqdm(range(self._length), desc="DALI preload"):
            frames.append(self[i])

        # Stack into single tensor
        self._frames_tensor = torch.stack(frames)
        print(f"[DALI] Preloaded: {self._frames_tensor.shape}, VRAM: {self._frames_tensor.numel()*4/1024/1024:.1f}MB")
        return self._frames_tensor

    def cleanup(self):
        """Delete temp H.265 file"""
        if hasattr(self, 'h265_path') and os.path.exists(self.h265_path):
            os.unlink(self.h265_path)
            print(f"[DALI] Cleaned up temp file")


class DALIVideoLoader(DALIFrameStreamer):
    """Alias for backward compatibility"""
    pass


class DALIChunkedStreamer:
    """
    SYNCHRONOUS DALI streamer - NO threads, NO queues, NO deadlocks.
    Load chunks on-demand, cache for reuse. 0% duplicates guaranteed.
    """

    def __init__(self, video_path, chunk_size=500, device_id=0, image_size=1024, max_height=720):
        if not HAS_DALI:
            raise RuntimeError("DALI not installed! pip install nvidia-dali-cuda120")

        self.chunk_size = chunk_size
        self.device_id = device_id
        self.image_size = image_size
        self.max_height = max_height

        # SAM2 normalization constants
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(device_id)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(device_id)

        # Convert to H.265 for GPU decoding, get ORIGINAL frame count
        self.h265_path, self._length = self._convert_to_h265(video_path)

        # Build DALI pipeline
        self._build_pipeline()

        # Sequential chunk tracking (DALI reads forward only)
        self.next_chunk_num = 0

        # Chunk cache with LRU eviction
        self.chunk_cache = {}
        self.cache_order = []
        self.max_cached_chunks = 3

        print(f"[DALI-SYNC] {self._length} frames, {chunk_size}-frame chunks")
        print(f"[DALI-SYNC] Synchronous on-demand loading (NO background threads)")

    def _convert_to_h265(self, input_path):
        """Convert video to DALI-friendly H.265 stream. Returns (path, orig_frame_count)"""
        import tempfile

        cap = cv2.VideoCapture(input_path)
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if orig_height > self.max_height:
            scale = self.max_height / orig_height
            new_width = int(orig_width * scale) // 2 * 2
            new_height = self.max_height // 2 * 2
        else:
            new_width, new_height = orig_width, orig_height

        self.video_height = new_height
        self.video_width = new_width

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()

        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx265', '-crf', '18', '-preset', 'ultrafast',
            '-vf', f'scale={new_width}:{new_height}',
            '-pix_fmt', 'yuv420p',
            '-an',
            temp_path
        ]

        print(f"[DALI-SYNC] Converting to H.265: {orig_width}x{orig_height} -> {new_width}x{new_height}")
        subprocess.run(cmd, capture_output=True, check=True)

        size_mb = os.path.getsize(temp_path) / (1024**2)
        print(f"[DALI-SYNC] H.265 video: {size_mb:.1f}MB, {orig_frames} frames")

        return temp_path, orig_frames

    def _build_pipeline(self):
        """Build DALI pipeline - batch loading for speed"""
        h265_path = self.h265_path
        device_id = self.device_id
        chunk_size = self.chunk_size

        @pipeline_def
        def video_pipe():
            video = fn.readers.video(
                device="gpu",
                filenames=[h265_path],
                sequence_length=chunk_size,  # Full chunk per run - FAST!
                stride=1,                     # Consecutive frames
                step=chunk_size,              # Non-overlapping chunks
                random_shuffle=False,
                image_type=types.RGB,
                dtype=types.UINT8,
                pad_sequences=True,           # Pad last chunk if needed
                name="video_reader"
            )
            return video

        self.pipe = video_pipe(
            batch_size=1,
            num_threads=4,
            device_id=device_id,
            prefetch_queue_depth=2
        )
        self.pipe.build()

    def _load_chunk(self, chunk_num):
        """Load chunk synchronously - fast batch loading"""
        chunk_start = chunk_num * self.chunk_size

        # Skip chunks if we're behind (DALI reads forward only)
        while self.next_chunk_num < chunk_num:
            self.pipe.run()  # Discard
            self.next_chunk_num += 1

        # Load this chunk (full chunk in one run!)
        output = self.pipe.run()
        chunk_tensor = torch.from_dlpack(output[0].as_tensor()).squeeze(0)  # [chunk_size, H, W, C]
        self.next_chunk_num += 1

        # Trim last chunk if padded
        actual_size = min(self.chunk_size, self._length - chunk_start)
        if chunk_tensor.shape[0] > actual_size:
            chunk_tensor = chunk_tensor[:actual_size]

        # Cache with LRU eviction
        self.chunk_cache[chunk_num] = chunk_tensor
        self.cache_order.append(chunk_num)

        if len(self.chunk_cache) > self.max_cached_chunks:
            oldest = self.cache_order.pop(0)
            if oldest in self.chunk_cache and oldest != chunk_num:
                del self.chunk_cache[oldest]

        print(f"[DALI-SYNC] Loaded chunk {chunk_num}: frames {chunk_start}-{chunk_start+len(chunk_tensor)-1}")
        return chunk_tensor

    def _get_chunk(self, chunk_num):
        """Get chunk from cache or load it"""
        if chunk_num in self.chunk_cache:
            return self.chunk_cache[chunk_num]
        return self._load_chunk(chunk_num)

    def __getitem__(self, idx):
        """Get frame - loads chunk on first access, cached for reuse"""
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Frame {idx} out of bounds (0-{self._length-1})")

        chunk_num = idx // self.chunk_size
        chunk = self._get_chunk(chunk_num)

        chunk_idx = idx - (chunk_num * self.chunk_size)
        frame = chunk[chunk_idx]  # [H, W, C] uint8

        # Process single frame
        frame = frame.float() / 255.0
        frame = frame.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        frame = torch.nn.functional.interpolate(
            frame,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        )
        frame = (frame - self.img_mean) / self.img_std
        return frame.squeeze(0)  # [C, H, W]

    def __len__(self):
        return self._length

    def cleanup(self):
        """Cleanup temp files"""
        self.chunk_cache.clear()
        if hasattr(self, 'h265_path') and os.path.exists(self.h265_path):
            try:
                os.unlink(self.h265_path)
                print(f"[DALI-SYNC] Cleaned up temp file")
            except:
                pass


class RevolutionMaskCompressor:
    """
    Ultra-efficient mask compression: Delta + RLE + zlib.
    Achieves 100x compression vs raw PNGs for typical masks.
    """

    def __init__(self):
        import zlib
        self.prev_mask = None
        self.frame_count = 0
        self.compressed_masks = []

    def compress_mask(self, mask_tensor):
        """Compress mask using delta + RLE + zlib"""
        import zlib
        from itertools import groupby

        # Convert to numpy bool
        if torch.is_tensor(mask_tensor):
            mask = mask_tensor.cpu().numpy().astype(bool)
        else:
            mask = mask_tensor.astype(bool)

        if self.frame_count == 0 or self.prev_mask is None:
            # First frame: store full mask
            data = mask
            is_delta = False
        else:
            # Delta: XOR with previous (most bits will be 0)
            data = np.logical_xor(mask, self.prev_mask)
            is_delta = True

        self.prev_mask = mask.copy()
        self.frame_count += 1

        # RLE encode
        rle = self._rle_encode(data.flatten())

        # zlib compress
        compressed = zlib.compress(rle.tobytes(), level=6)

        # Store with metadata
        self.compressed_masks.append({
            'is_delta': is_delta,
            'shape': mask.shape,
            'data': compressed
        })

        return len(compressed)

    def _rle_encode(self, flat_array):
        """Run-length encoding for binary array"""
        from itertools import groupby
        rle = []
        for val, group in groupby(flat_array):
            length = sum(1 for _ in group)
            rle.extend([int(val), length])
        return np.array(rle, dtype=np.uint32)

    def save_to_file(self, output_path):
        """Save all compressed masks to a single file"""
        import pickle
        import zlib

        # Serialize and compress entire structure
        data = pickle.dumps(self.compressed_masks)
        compressed = zlib.compress(data, level=9)

        with open(output_path, 'wb') as f:
            f.write(compressed)

        size_kb = len(compressed) / 1024
        print(f"[MaskCompress] Saved {self.frame_count} masks to {size_kb:.1f}KB")
        return output_path

    @staticmethod
    def load_from_file(input_path):
        """Load and decompress masks from file"""
        import pickle
        import zlib
        from itertools import groupby

        with open(input_path, 'rb') as f:
            compressed = f.read()

        data = pickle.loads(zlib.decompress(compressed))

        masks = []
        prev_mask = None

        for entry in data:
            # Decompress RLE
            rle_data = np.frombuffer(zlib.decompress(entry['data']), dtype=np.uint32)

            # Decode RLE
            flat = []
            for i in range(0, len(rle_data), 2):
                val, length = rle_data[i], rle_data[i+1]
                flat.extend([bool(val)] * length)

            mask = np.array(flat, dtype=bool).reshape(entry['shape'])

            # Apply delta if needed
            if entry['is_delta'] and prev_mask is not None:
                mask = np.logical_xor(mask, prev_mask)

            prev_mask = mask.copy()
            masks.append((mask * 255).astype(np.uint8))

        return masks


class YUV420FrameStorage:
    """
    Load frames from disk and compress to YUV420 in SINGLE PASS.
    No double loading - compress as we load!
    """

    def __init__(self, frames_dir, image_size=1024, device="cuda"):
        import torch.nn.functional as F
        from PIL import Image
        import torchvision.transforms as transforms

        self.device = device
        self.compressed = []
        self.video_height = None
        self.video_width = None

        # SAM2's normalization constants
        img_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        img_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        # Get frame paths
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        # SAM2's transform (resize to internal size)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        print(f"[YUV420] Loading & compressing {len(frame_files)} frames (single pass)...", flush=True)
        for fname in tqdm(frame_files, desc="YUV420 load+compress"):
            img = Image.open(os.path.join(frames_dir, fname)).convert("RGB")

            # Store original dimensions from first frame
            if self.video_height is None:
                self.video_width, self.video_height = img.size

            # Transform to tensor and normalize (like SAM2 does)
            tensor = transform(img).to(device)  # [3, H, W]
            tensor = (tensor - img_mean) / img_std

            # Compress immediately to YUV420
            compressed = self._compress(tensor)
            self.compressed.append(compressed)

            del tensor, img

        # Stats
        orig_size = len(self.compressed) * 3 * image_size * image_size * 4
        comp_size = sum(c['y'].numel() + c['u'].numel() + c['v'].numel() for c in self.compressed) * 2
        print(f"[YUV420] Done! {orig_size/1e9:.2f}GB -> {comp_size/1e9:.2f}GB ({orig_size/comp_size:.1f}:1)", flush=True)
        torch.cuda.empty_cache()

    def _compress(self, frame):
        """Compress [3, H, W] tensor to YUV420"""
        import torch.nn.functional as F

        frame = frame.to(self.device)
        r, g, b = frame[0], frame[1], frame[2]

        # RGB to YUV
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b

        # 4:2:0 subsampling - U/V at quarter resolution
        u_sub = F.avg_pool2d(u.unsqueeze(0).unsqueeze(0), 2).squeeze()
        v_sub = F.avg_pool2d(v.unsqueeze(0).unsqueeze(0), 2).squeeze()

        # Store as half precision
        return {
            'y': y.half(),
            'u': u_sub.half(),
            'v': v_sub.half()
        }

    def _decompress(self, compressed):
        """Decompress YUV420 back to [3, H, W] tensor"""
        import torch.nn.functional as F

        y = compressed['y'].float()
        u = F.interpolate(compressed['u'].unsqueeze(0).unsqueeze(0).float(),
                         scale_factor=2, mode='bilinear', align_corners=False).squeeze()
        v = F.interpolate(compressed['v'].unsqueeze(0).unsqueeze(0).float(),
                         scale_factor=2, mode='bilinear', align_corners=False).squeeze()

        # YUV to RGB
        r = y + 1.140 * v
        g = y - 0.395 * u - 0.581 * v
        b = y + 2.032 * u

        return torch.stack([r, g, b], dim=0)

    def __getitem__(self, index):
        return self._decompress(self.compressed[index])

    def __len__(self):
        return len(self.compressed)


class YUV420FrameStorageReordered:
    """
    Wrapper that provides frames from an existing YUV420FrameStorage in a different order.
    Used for backward tracking by reversing frame order.
    """

    def __init__(self, original_storage, frame_indices):
        """
        Args:
            original_storage: YUV420FrameStorage instance
            frame_indices: List of original frame indices in desired order
                           e.g., [100, 99, 98, ..., 0] for backward from frame 100
        """
        self.original = original_storage
        self.frame_indices = frame_indices
        self.video_height = original_storage.video_height
        self.video_width = original_storage.video_width

    def __getitem__(self, index):
        # Map new index to original frame index
        original_idx = self.frame_indices[index]
        return self.original[original_idx]

    def __len__(self):
        return len(self.frame_indices)


def init_state_reordered(predictor, original_yuv_frames, frame_indices, device="cuda"):
    """
    Initialize SAM2 inference state with reordered frames (for backward tracking).
    Reuses existing YUV420 compressed frames but accesses them in different order.
    """
    from collections import OrderedDict

    # Wrap existing frames with new order
    reordered_frames = YUV420FrameStorageReordered(original_yuv_frames, frame_indices)

    # Build inference_state
    inference_state = {}
    inference_state["images"] = reordered_frames
    inference_state["num_frames"] = len(reordered_frames)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = reordered_frames.video_height
    inference_state["video_width"] = reordered_frames.video_width
    inference_state["device"] = torch.device(device)
    inference_state["storage_device"] = torch.device(device)

    # Initialize tracking structures
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up visual backbone on frame 0 (which is original frame_idx_start)
    print("[SAM2-Reversed] Warming up visual backbone on reversed sequence...", flush=True)
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def init_state_yuv420(predictor, frames_dir, device="cuda"):
    """
    Initialize SAM2 inference state with YUV420 compressed frames.
    SKIPS SAM2's load_video_frames() entirely - no double loading!
    """
    from collections import OrderedDict

    # Load frames with YUV420 compression (single pass)
    yuv_frames = YUV420FrameStorage(frames_dir, image_size=predictor.image_size, device=device)

    # Build inference_state manually (same structure as SAM2's init_state)
    inference_state = {}
    inference_state["images"] = yuv_frames
    inference_state["num_frames"] = len(yuv_frames)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = yuv_frames.video_height
    inference_state["video_width"] = yuv_frames.video_width
    inference_state["device"] = torch.device(device)
    inference_state["storage_device"] = torch.device(device)

    # Initialize tracking structures (same as SAM2's init_state)
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Warm up visual backbone on frame 0
    print("[SAM2-YUV420] Warming up visual backbone...", flush=True)
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def init_state_dali(predictor, video_path, device="cuda", max_height=720, chunk_size=500):
    """
    Initialize SAM2 inference state with DALI chunked GPU streaming.
    ZERO CPU RAM - frames load directly from disk to GPU in chunks!

    Args:
        chunk_size: Frames per chunk (500 = ~1.3GB VRAM per chunk)
    """
    from collections import OrderedDict

    if not HAS_DALI:
        raise RuntimeError("DALI not available! pip install nvidia-dali-cuda120")

    # Create DALI CHUNKED streamer - O(1) within-chunk access, no OOM
    dali_streamer = DALIChunkedStreamer(
        video_path,
        chunk_size=chunk_size,
        device_id=0,
        image_size=predictor.image_size,
        max_height=max_height
    )

    # Build inference_state manually (same structure as SAM2's init_state)
    inference_state = {}
    inference_state["images"] = dali_streamer  # DALI replaces YUV420FrameStorage
    inference_state["num_frames"] = len(dali_streamer)
    inference_state["offload_video_to_cpu"] = False
    inference_state["offload_state_to_cpu"] = False
    inference_state["video_height"] = dali_streamer.video_height
    inference_state["video_width"] = dali_streamer.video_width
    inference_state["device"] = torch.device(device)
    inference_state["storage_device"] = torch.device(device)

    # Initialize tracking structures (same as SAM2's init_state)
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    inference_state["cached_features"] = {}
    inference_state["constants"] = {}
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    inference_state["output_dict_per_obj"] = {}
    inference_state["temp_output_dict_per_obj"] = {}
    inference_state["frames_tracked_per_obj"] = {}

    # Store reference for cleanup
    inference_state["_dali_streamer"] = dali_streamer

    # Warm up visual backbone on frame 0
    print("[SAM2-DALI] Warming up visual backbone (GPU-direct)...", flush=True)
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)

    return inference_state


def rle_encode_mask(mask_uint8):
    """RLE compress binary mask (50:1 compression for masks)"""
    binary = (mask_uint8 > 127).astype(np.uint8).flatten()
    if len(binary) == 0:
        return [], mask_uint8.shape

    runs = []
    current_val = binary[0]
    current_len = 1

    for i in range(1, len(binary)):
        if binary[i] == current_val:
            current_len += 1
        else:
            runs.append(current_len if current_val == 1 else -current_len)
            current_val = binary[i]
            current_len = 1
    runs.append(current_len if current_val == 1 else -current_len)

    return runs, mask_uint8.shape


def rle_decode_mask(rle_runs, shape):
    """Decompress RLE back to binary mask"""
    if not rle_runs:
        return np.zeros(shape, dtype=np.uint8)

    flat = []
    for run in rle_runs:
        flat.extend([1 if run > 0 else 0] * abs(run))

    return (np.array(flat, dtype=np.uint8).reshape(shape) * 255)


# Configuration
SAM2_CHECKPOINT = SAM2_PATH / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# TensorRT engines - WSL2 paths
ENCODER_ENGINE = "/mnt/d/watermarkz/sam2_trt_inference/engines_wsl2/sam2_encoder_fp16.engine"
DECODER_ENGINE = "/mnt/d/watermarkz/sam2_trt_inference/engines_wsl2/sam2_decoder_fp16_dynamic.engine"


def enable_optimizations():
    """Enable PyTorch optimizations for WSL2"""
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_properties(0).major

        # TF32 on Ampere+
        if compute_capability >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # cuDNN benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Memory allocator
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Inductor - DISABLED for faster testing
        # torch._inductor.config.coordinate_descent_tuning = True
        # torch._inductor.config.triton.unique_kernel_names = True
        # torch._inductor.config.fx_graph_cache = True

        # SDPA
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Disable gradients
        torch.set_grad_enabled(False)

        # Async CUDA
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def extract_frames(video_path, output_dir, max_height=720):
    """Extract frames from video, resized to 720p for faster loading"""
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate resize dimensions (maintain aspect ratio)
    if orig_height > max_height:
        scale = max_height / orig_height
        new_width = int(orig_width * scale)
        new_height = max_height
        print(f"[SAM2] Resizing {orig_width}x{orig_height} -> {new_width}x{new_height} (faster loading)")
    else:
        new_width, new_height = orig_width, orig_height

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize if needed for faster loading
        if orig_height > max_height:
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(f"{output_dir}/{frame_idx:05d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_idx += 1

    cap.release()
    print(f"[SAM2] Extracted {frame_idx} frames @ {fps:.1f} fps")
    return frame_idx, fps, orig_width, orig_height, new_width, new_height


def release_old_frames_complete(inference_state, keep_last=64):
    """
    Memory cleanup for SAM2 - SAFE version.

    IMPORTANT: Cannot truncate images tensor during propagation because
    propagate_in_video generator uses original frame indices.
    Only clean output_dict entries for memory savings.
    """
    try:
        # DON'T touch images tensor - generator uses original indices!
        # Only clean non_cond_frame_outputs to save some memory

        if "output_dict" in inference_state:
            output_dict = inference_state["output_dict"]

            # Only clean non-conditioning frame outputs (keep last N)
            if "non_cond_frame_outputs" in output_dict:
                keys = sorted(output_dict["non_cond_frame_outputs"].keys())
                if len(keys) > keep_last:
                    for old_idx in keys[:-keep_last]:
                        output_dict["non_cond_frame_outputs"].pop(old_idx, None)

            # DO NOT touch cond_frame_outputs - SAM2 needs them

        # Clean per-object output dictionaries
        if "output_dict_per_obj" in inference_state:
            for obj_id, obj_dict in inference_state["output_dict_per_obj"].items():
                if "non_cond_frame_outputs" in obj_dict:
                    keys = sorted(obj_dict["non_cond_frame_outputs"].keys())
                    if len(keys) > keep_last:
                        for old_idx in keys[:-keep_last]:
                            obj_dict["non_cond_frame_outputs"].pop(old_idx, None)

        # Force cleanup
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[SAM2-Memory] Cleanup warning: {e}")


def load_frames_as_tensors(frames_dir, start_idx, end_idx, img_mean, img_std, image_size):
    """
    Load frames from disk and convert to tensors matching SAM2's format.

    Returns list of tensors ready to extend inference_state["images"]
    """
    from PIL import Image
    import torchvision.transforms as transforms

    # SAM2 uses this transform internally
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    images = []
    for i in range(start_idx, end_idx):
        frame_path = os.path.join(frames_dir, f"{i:05d}.jpg")
        if os.path.exists(frame_path):
            img = Image.open(frame_path).convert("RGB")
            img_tensor = transform(img)
            images.append(img_tensor)

    return images


def track_video_pytorch_only(frames_dir, total_frames, output_masks_dir, points=None, labels=None, bbox=None, frame_idx_start=0, video_path=None):
    """
    SAM2 video tracking using OFFICIAL memory management parameters.

    Uses start_frame_idx and max_frame_num_to_track for chunked processing
    instead of manual tensor manipulation.

    Args:
        points: List of (x, y) tuples for multiple click points
        labels: List of labels (1=foreground, 0=background) for each point
        video_path: Original video path for DALI streaming (optional)
    """
    import gc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_optimizations()

    # Check if DALI streaming is enabled
    USE_DALI_STREAMING = os.getenv('USE_DALI_STREAMING', '0').lower() in ('1', 'true', 'yes', 'on')
    SAM2_MAX_HEIGHT = int(os.getenv('SAM2_MAX_HEIGHT', '720'))

    # Create output directory
    os.makedirs(output_masks_dir, exist_ok=True)

    CHUNK_SIZE = 128  # Process 128 frames at a time (memory optimized for long videos)

    # Build predictor
    print(f"[SAM2-Official] Building predictor with official memory management...")
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        str(SAM2_CHECKPOINT),
        device=device,
        vos_optimized=False
    )
    torch.cuda.empty_cache()

    inference_state = None  # Initialize for finally block

    try:
        # Choose initialization method: DALI (GPU-direct) or YUV420 (CPU RAM)
        if USE_DALI_STREAMING and HAS_DALI and video_path:
            print(f"[SAM2-DALI] GPU-direct streaming enabled! ZERO CPU RAM!")
            print(f"[SAM2-DALI] Video: {video_path}, Max height: {SAM2_MAX_HEIGHT}")
            inference_state = init_state_dali(predictor, video_path, device=device, max_height=SAM2_MAX_HEIGHT)
            actual_frames = inference_state['num_frames']
            print(f"[SAM2-DALI] Ready! {actual_frames} frames streaming from GPU", flush=True)
        else:
            if USE_DALI_STREAMING and not HAS_DALI:
                print(f"[SAM2-DALI] WARNING: DALI requested but not installed! Falling back to YUV420")
            if USE_DALI_STREAMING and not video_path:
                print(f"[SAM2-DALI] WARNING: DALI requested but no video_path! Falling back to YUV420")
            # Initialize state with YUV420 compression - SKIPS SAM2's frame loading entirely!
            print(f"[SAM2-YUV420] Initializing with {total_frames} frames (single pass, no double loading)...")
            inference_state = init_state_yuv420(predictor, frames_dir, device=device)
            actual_frames = inference_state['num_frames']
            print(f"[SAM2-YUV420] Ready! {actual_frames} frames loaded", flush=True)

        # Clamp frame_idx_start to valid range (frontend may calculate wrong index if FPS differs)
        if frame_idx_start < 0 or frame_idx_start >= actual_frames:
            old_idx = frame_idx_start
            frame_idx_start = max(0, min(frame_idx_start, actual_frames - 1))
            print(f"[SAM2] WARNING: frame_idx_start {old_idx} out of range, clamped to {frame_idx_start}")

        # Add initial prompt
        if bbox is not None:
            box_array = np.array(bbox, dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx_start,
                obj_id=1,
                box=box_array
            )
            print(f"[SAM2-Official] Added bbox prompt at frame {frame_idx_start}")
        elif points is not None and len(points) > 0:
            points_array = np.array([[p[0], p[1]] for p in points], dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32) if labels else np.ones(len(points), dtype=np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx_start,
                obj_id=1,
                points=points_array,
                labels=labels_array
            )
            print(f"[SAM2-Official] Added point prompt at frame {frame_idx_start}")

        masks_saved = 0

        # BIDIRECTIONAL PROPAGATION - simple internal reverse=True
        print(f"\n[SAM2-Bidirectional] Starting from frame {frame_idx_start}, propagating both directions")

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):

                # ============================================================
                # BACKWARD: frame_idx_start -> 0 (using internal reverse=True)
                # ============================================================
                backward_frames = frame_idx_start + 1  # Include click frame
                if frame_idx_start > 0:
                    print(f"\n[SAM2-Backward] Propagating frames {frame_idx_start} -> 0 ({backward_frames} frames)")

                    for frame_idx, obj_ids, mask_logits in tqdm(
                        predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=frame_idx_start,
                            max_frame_num_to_track=backward_frames,
                            reverse=True
                        ),
                        total=backward_frames,
                        desc="Backward"
                    ):
                        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                        mask_uint8 = (mask * 255).astype(np.uint8)

                        if np.sum(mask_uint8 > 127) < 100:
                            mask_uint8[:] = 0

                        cv2.imwrite(f"{output_masks_dir}/{frame_idx:05d}.png", mask_uint8)
                        masks_saved += 1

                        del mask_logits, mask

                    print(f"[SAM2-Backward] Complete: {backward_frames} masks")
                    gc.collect()
                    torch.cuda.empty_cache()

                # ============================================================
                # FORWARD: frame_idx_start -> end
                # ============================================================
                forward_frames = actual_frames - frame_idx_start
                if forward_frames > 0:
                    if frame_idx_start > 0:
                        print(f"\n[SAM2-Forward] Propagating frames {frame_idx_start} -> {actual_frames - 1} ({forward_frames} frames)")
                    else:
                        print(f"\n[SAM2-Forward] Propagating frames 0 -> {actual_frames - 1} ({forward_frames} frames)")

                    for frame_idx, obj_ids, mask_logits in tqdm(
                        predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=frame_idx_start,
                            max_frame_num_to_track=forward_frames,
                            reverse=False
                        ),
                        total=forward_frames,
                        desc="Forward"
                    ):
                        # Skip click frame if already saved by backward
                        if frame_idx_start > 0 and frame_idx == frame_idx_start:
                            continue

                        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                        mask_uint8 = (mask * 255).astype(np.uint8)

                        if np.sum(mask_uint8 > 127) < 100:
                            mask_uint8[:] = 0

                        cv2.imwrite(f"{output_masks_dir}/{frame_idx:05d}.png", mask_uint8)
                        masks_saved += 1

                        del mask_logits, mask

                    print(f"[SAM2-Forward] Complete: {forward_frames} masks")
                    gc.collect()
                    torch.cuda.empty_cache()

        print(f"[SAM2] Complete! Saved {masks_saved} masks to {output_masks_dir}")
        return masks_saved

    finally:
        # Cleanup DALI streamer if used (deletes temp H.265 file)
        if inference_state is not None and "_dali_streamer" in inference_state:
            inference_state["_dali_streamer"].cleanup()
        # Always cleanup predictor
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def convert_windows_to_wsl_path(path):
    """Convert Windows path to WSL2 path if needed"""
    if not path:
        return path
    # Handle D: or d: drive
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        return f'/mnt/{drive}' + path[2:].replace('\\', '/')
    elif '\\' in path:
        return path.replace('\\', '/')
    return path


def main():
    parser = argparse.ArgumentParser(description='SAM2 Video Tracking for WSL2')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('output_masks_dir', help='Output directory for masks')

    # Prompt options - mutually exclusive
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument('--point', help='Point coordinates as x,y')
    prompt_group.add_argument('--bbox', help='Bounding box as x1,y1,x2,y2')

    parser.add_argument('--frame-idx', type=int, default=0, help='Frame index to start tracking from')
    parser.add_argument('--temp-frames-dir', default='/tmp/sam2_frames', help='Temp dir for frames')

    args = parser.parse_args()

    # Parse prompt
    point = None
    bbox = None

    if args.point:
        try:
            x, y = map(int, args.point.split(','))
            point = (x, y)
            print(f"[SAM2] Using point prompt: {point}")
        except:
            print(f"[ERROR] Invalid point format: {args.point}. Use x,y (e.g., 200,150)")
            sys.exit(1)
    elif args.bbox:
        try:
            coords = list(map(int, args.bbox.split(',')))
            if len(coords) != 4:
                raise ValueError("Need exactly 4 coordinates")
            bbox = coords  # [x1, y1, x2, y2]
            print(f"[SAM2] Using bbox prompt: {bbox}")
        except:
            print(f"[ERROR] Invalid bbox format: {args.bbox}. Use x1,y1,x2,y2 (e.g., 100,100,300,300)")
            sys.exit(1)

    # Convert Windows paths to WSL2 paths
    video_path = convert_windows_to_wsl_path(args.video_path)
    output_masks_dir = convert_windows_to_wsl_path(args.output_masks_dir)
    temp_frames_dir = convert_windows_to_wsl_path(args.temp_frames_dir)

    print(f"[SAM2] Video: {video_path}")
    print(f"[SAM2] Output: {output_masks_dir}")
    print(f"[SAM2] Frame idx: {args.frame_idx}")

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # Check if DALI streaming is enabled
    USE_DALI_STREAMING = os.getenv('USE_DALI_STREAMING', '0').lower() in ('1', 'true', 'yes', 'on')
    SAM2_MAX_HEIGHT = int(os.getenv('SAM2_MAX_HEIGHT', '720'))

    if USE_DALI_STREAMING and HAS_DALI:
        # DALI MODE: Skip frame extraction - stream directly from video!
        print(f"[SAM2-DALI] GPU-direct streaming mode - SKIPPING frame extraction!")

        # Get video info for coordinate scaling
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Calculate target dimensions (same logic as DALI)
        if orig_h > SAM2_MAX_HEIGHT:
            scale_factor = SAM2_MAX_HEIGHT / orig_h
            new_w = int(orig_w * scale_factor) // 2 * 2
            new_h = SAM2_MAX_HEIGHT // 2 * 2
        else:
            new_w, new_h = orig_w, orig_h

        print(f"[SAM2-DALI] Video: {total_frames} frames @ {fps:.1f}fps, {orig_w}x{orig_h} -> {new_w}x{new_h}")
    else:
        # YUV420 MODE: Extract frames to disk
        total_frames, fps, orig_w, orig_h, new_w, new_h = extract_frames(video_path, temp_frames_dir)

    # Scale coordinates if frames were resized
    scale = new_h / orig_h if orig_h != new_h else 1.0
    if scale != 1.0:
        print(f"[SAM2] Scaling coordinates by {scale:.3f} (original {orig_w}x{orig_h} -> {new_w}x{new_h})")

        if point is not None:
            point = (int(point[0] * scale), int(point[1] * scale))
            print(f"[SAM2] Scaled point: {point}")

        if bbox is not None:
            bbox = [int(c * scale) for c in bbox]
            print(f"[SAM2] Scaled bbox: {bbox}")

    # Run tracking (pure PyTorch - simpler and still fast)
    # Convert single point to list format expected by function
    points_list = [point] if point else None
    labels_list = [1] if point else None  # 1 = foreground

    masks_saved = track_video_pytorch_only(
        temp_frames_dir,
        total_frames,
        output_masks_dir,
        points=points_list,
        labels=labels_list,
        bbox=bbox,
        frame_idx_start=args.frame_idx,
        video_path=video_path  # Pass for DALI streaming
    )

    # Output JSON for Windows to read
    result = {
        'status': 'success',
        'total_frames': total_frames,
        'masks_saved': masks_saved,
        'fps': fps,
        'masks_dir': output_masks_dir
    }
    print(f"\n[RESULT] {json.dumps(result)}")


if __name__ == '__main__':
    main()
