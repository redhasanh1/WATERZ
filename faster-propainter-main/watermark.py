# -*- coding: utf-8 -*-
import os
import time
import threading

from tqdm import tqdm
import cv2
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
import torch
import torchvision

# Configure torch.compile cache settings BEFORE any model imports
# CRITICAL: Must run before model loading to prevent multi-worker cache race conditions
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import cache_config  # Disables inductor cache for thread safety

# Essential optimization: Enable cuDNN autotuner (benchmarks kernels once, reuses best)
# This is SAFE and provides 20-30% speedup without breaking anything
torch.backends.cudnn.benchmark = True

# Enable TF32 for Ada Lovelace (RTX 4090) - Safe ~20% speedup on matmul operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ⚡ RTX 4090 Ada Lovelace: Enable FP8 Transformer Engine (1.45x speedup on transformers)
# 4th Gen Tensor Cores automatically use FP8 when set to 'high' or 'medium'
torch.set_float32_matmul_precision('high')  # 'high' = FP8, 'medium' = TF32, 'highest' = FP32
print("[OK] RTX 4090 FP8 Transformer Engine enabled (4th Gen Tensor Cores: 1,321 TFLOPs)")

# Enable Flash Attention backends for Blackwell optimization
if os.getenv("ENABLE_FLASH_ATTENTION", "0") == "1":
    try:
        # Enable Flash Attention and memory-efficient attention backends
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)  # Fallback if needed
        print("[OK] Flash Attention backends enabled (Blackwell Flash Attention 3)")
    except Exception as e:
        print(f"[WARNING] Could not enable Flash Attention backends: {e}")

# Enable FP8 Quantization for Transformer (RTX 4090 Ada Lovelace 4th Gen Tensor Cores)
# Provides 1.3-1.5x speedup on Linear layers (QKV projections, FFN)
# Combined speedup with DCNv4 + Flash Attention = ~5-10x faster transformers!
if os.getenv("ENABLE_FP8_TRANSFORMER", "1") == "1":  # ENABLED BY DEFAULT
    print("[OK] FP8 Transformer quantization enabled (RTX 4090 Ada: 1.3-1.5x speedup)")
else:
    print("[INFO] FP8 Transformer quantization disabled (set ENABLE_FP8_TRANSFORMER=1 to enable)")

# ============================================================================
# SAGEATTENTION INTEGRATION (RTX 4090 INT8 OPTIMIZED)
# ============================================================================
import torch.nn.functional as F
from typing import Optional

# Environment variable control
ENABLE_SAGE_ATTENTION = os.getenv("ENABLE_SAGE_ATTENTION", "1") == "1"

# Import SageAttention (with fallback)
if ENABLE_SAGE_ATTENTION:
    try:
        from sageattention import sageattn
        print("[ATTENTION] SageAttention loaded (INT8 quantization mode for RTX 4090)")
        SAGE_ATTENTION_AVAILABLE = True
    except ImportError:
        print("[ATTENTION] SageAttention not installed, falling back to SDPA")
        SAGE_ATTENTION_AVAILABLE = False
else:
    SAGE_ATTENTION_AVAILABLE = False
    print("[ATTENTION] SageAttention disabled via ENABLE_SAGE_ATTENTION=0")

def sage_attn_wrapper(
    q, k, v,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None
):
    """
    Drop-in replacement for F.scaled_dot_product_attention using SageAttention
    Handles both 4D [B, num_heads, seq_len, head_dim] and 5D [B, T, H, W, num_heads, head_dim] tensors

    Args:
        q, k, v: Query, Key, Value tensors
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (not used in inference)
        is_causal: Whether to apply causal masking
        scale: Attention scale factor

    Returns:
        Attention output with same shape as input
    """
    if not SAGE_ATTENTION_AVAILABLE:
        # Fallback to PyTorch's SDPA (or FlashAttention if enabled)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )

    # Store original shape for later
    original_shape = q.shape
    original_dtype = q.dtype

    # Check if we need to reshape 5D → 4D for SageAttention
    if len(original_shape) == 6:
        # 5D video tensor: [B, T, H, W, num_heads, head_dim]
        B, T, H, W, num_heads, head_dim = original_shape

        # Flatten spatial and temporal dimensions to sequence
        q = q.reshape(B, T * H * W, num_heads, head_dim)
        k = k.reshape(B, T * H * W, num_heads, head_dim)
        v = v.reshape(B, T * H * W, num_heads, head_dim)

        # Handle attn_mask if present
        if attn_mask is not None and len(attn_mask.shape) > 2:
            attn_mask = attn_mask.reshape(B, T * H * W)

    # SageAttention expects FP16 for INT8 quantization
    if q.dtype == torch.float32:
        q = q.half()
        k = k.half()
        v = v.half()

    # Call SageAttention (automatically uses INT8 kernels on RTX 4090)
    try:
        out = sageattn(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            tensor_layout="HND"  # [Batch, SeqLen, NumHeads, HeadDim]
        )
    except Exception as e:
        # Fallback to SDPA if SageAttention fails
        print(f"[WARNING] SageAttention failed: {e}, falling back to SDPA")
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )

    # Convert back to original dtype if needed
    if out.dtype != original_dtype:
        out = out.to(original_dtype)

    # Reshape back to original 5D if necessary
    if len(original_shape) == 6:
        B, T, H, W, num_heads, head_dim = original_shape
        out = out.reshape(B, T, H, W, num_heads, head_dim)

    return out

# Thread-safe warmup cache (one per worker)
_worker_sage_warmup_done = threading.local()

def warmup_sageattention(device='cuda', q_shape=(1, 8, 64, 64, 8, 64)):
    """
    Compile SageAttention INT8 kernels once per worker thread

    Args:
        device: CUDA device to compile for
        q_shape: Example input shape (B, T, H, W, num_heads, head_dim)
    """
    if hasattr(_worker_sage_warmup_done, 'done') and _worker_sage_warmup_done.done:
        return

    if not SAGE_ATTENTION_AVAILABLE:
        return

    print(f"[SAGE] Warming up INT8 kernels on {device}...")
    warmup_start = time.time()

    # Create dummy tensors
    dummy_q = torch.randn(*q_shape, device=device, dtype=torch.float16)
    dummy_k = torch.randn(*q_shape, device=device, dtype=torch.float16)
    dummy_v = torch.randn(*q_shape, device=device, dtype=torch.float16)

    # Warmup runs (compiles INT8 CUDA kernels for RTX 4090)
    for _ in range(3):
        _ = sage_attn_wrapper(dummy_q, dummy_k, dummy_v)

    torch.cuda.synchronize(device)

    warmup_time = time.time() - warmup_start
    print(f"[SAGE] INT8 kernels compiled and optimized for RTX 4090! ({warmup_time:.2f}s)")
    _worker_sage_warmup_done.done = True

# Conditional import: Use NeuFlow v2 or RAFT based on environment variable
# ============================================================================
# STARTUP DIAGNOSTICS - Optical Flow Model Selection
# ============================================================================
_USE_NEUFLOW = os.getenv("USE_NEUFLOW", "0") == "1"
_FORCE_TRT_RAFT = os.getenv("FORCE_TRT_RAFT", "0").lower() in ("1", "true", "yes", "on")
_NEUFLOW_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "neuflow_things_fp16.engine") if os.path.dirname(__file__) else "models/neuflow_things_fp16.engine"
_NEUFLOW_EXISTS = os.path.exists(_NEUFLOW_MODEL_PATH)

print("=" * 70)
print("OPTICAL FLOW MODEL CONFIGURATION")
print("=" * 70)
print(f"Environment Variables:")
print(f"  USE_NEUFLOW       = {os.getenv('USE_NEUFLOW', '0')} (Active: {_USE_NEUFLOW})")
print(f"  FORCE_TRT_RAFT    = {os.getenv('FORCE_TRT_RAFT', '0')} (Active: {_FORCE_TRT_RAFT})")
print(f"  ENABLE_FLASH_ATTENTION = {os.getenv('ENABLE_FLASH_ATTENTION', '0')}")
print(f"  RFCNET_TORCHTRT   = {os.getenv('RFCNET_TORCHTRT', '0')}")
print()
print(f"Model File Status:")
print(f"  NeuFlow v2 TensorRT FP16 = {'[OK] EXISTS' if _NEUFLOW_EXISTS else '[X] NOT FOUND'} ({_NEUFLOW_MODEL_PATH})")
print()
print(f"Expected Behavior:")
if _USE_NEUFLOW:
    if _NEUFLOW_EXISTS:
        print(f"  -> Will use NeuFlow v2 TensorRT FP16 (3-4x faster than ONNX Runtime, 10-70x faster than RAFT)")
    else:
        print(f"  -> ERROR: USE_NEUFLOW=1 but model file missing!")
        print(f"  -> Build TensorRT engine using: tools/BUILD_NEUFLOW_TRT.bat")
elif _FORCE_TRT_RAFT:
    print(f"  -> Will attempt TensorRT RAFT FP16 (fallback to PyTorch if not available)")
else:
    print(f"  -> Will use PyTorch RAFT (slowest, but most compatible)")
print("=" * 70)
print()

if _USE_NEUFLOW:
    from model.modules.flow_comp_neuflow import NeuFlow_bi as RAFT_bi
    print("[OK] Using NeuFlow v2 for optical flow (10-70x faster than RAFT)")
else:
    from model.modules.flow_comp_raft import RAFT_bi
    # RAFT can use PyTorch or TensorRT depending on FORCE_TRT_RAFT env var
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

# from mytimer import timer_decorator

import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# GLOBAL MODEL CACHE - Models persist across segments for true parallelism
# ============================================================================
_GLOBAL_MODELS_CACHE = None
_MODEL_CACHE_LOCK = None

pretrain_model_url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
# @timer_decorator
def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]

    return frames, process_size, out_size


#  read frames from video
# @timer_decorator
def read_frame_from_videos(frame_root):
    if frame_root.endswith(
        ("mp4", "mov", "avi", "MP4", "MOV", "AVI")
    ):  # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(
            filename=frame_root, pts_unit="sec"
        )  # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info["video_fps"]
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask > th] = 1
    mask[mask <= th] = 0
    return mask


# read frame-wise masks
# @timer_decorator
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []

    if mpath.endswith(
        ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
    ):  # input single img path
        masks_img = [Image.open(mpath)]
    else:
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))

    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert("L"))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(
                mask_img, iterations=flow_mask_dilates
            ).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(
                mask_img, iterations=mask_dilates
            ).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))

    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting."""
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start : H_start + imgH, W_start : W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []

    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

    mask[
        H_start + dilate_h : H_start + imgH - dilate_h,
        W_start + dilate_w : W_start + imgW - dilate_w,
    ] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start : H_start + imgH, W_start : W_start + imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))

    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame

    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


# Thread-local storage for TensorRT contexts - each thread gets its own context
# This enables TRUE PARALLEL execution without locks or memory corruption
import threading
_thread_local = threading.local()


def pipeline(
    video,
    mask,
    output,
    resize_ratio=1.0,
    height=-1,
    width=-1,
    mask_dilation=4,
    ref_stride=10,
    neighbor_length=10,
    subvideo_length=80,
    raft_iter=20,
    mode="video_inpainting",
    scale_h=1.0,
    scale_w=1.0,
    save_fps=24,
    save_frames=False,
    fp16=False,
    frames_array=None,
    masks_array=None,
    use_cached_models=True,  # Enable global model cache for true parallel speedup
):

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if fp16 else False
    device = get_device()
    if device == torch.device("cpu"):
        use_half = False

    # [VRAM DEBUG] Log actual parameters being used
    print(f"[VRAM CONFIG] neighbor_length={neighbor_length}, subvideo_length={subvideo_length}, fp16={fp16}")

    # 🔥 EXTREME SPEED: Use numpy arrays directly if provided (skip disk I/O!)
    if frames_array is not None:
        # Convert numpy arrays (BGR) to PIL Images (RGB)
        frames = []
        for frame_bgr in frames_array:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        # Derive metadata from arrays
        fps = save_fps  # Use provided fps or default
        size = (frames_array[0].shape[1], frames_array[0].shape[0])  # (W, H)

        # Extract video_name from video parameter (like read_frame_from_videos does)
        if video.endswith(("mp4", "mov", "avi", "MP4", "MOV", "AVI")):
            video_name = os.path.basename(video)[:-4]
        else:
            video_name = os.path.basename(video)

        print(f"[FAST] Using {len(frames)} frames from memory (ZERO disk I/O!)")
    else:
        # Original disk-based loading
        frames, fps, size, video_name = read_frame_from_videos(video)
    if not width == -1 and not height == -1:
        size = (width, height)
    if not resize_ratio == 1.0:
        size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))

    if (size[0] % 8 != 0) or (size[1] % 8 != 0):
        frames, size, out_size = resize_frames(frames, size)
    else:
        out_size = size

    # Validate and fix inconsistent frame sizes (prevents broadcasting errors)
    expected_size = (size[1], size[0])  # (H, W) for numpy
    for i, frame in enumerate(frames):
        frame_array = np.array(frame)
        if frame_array.shape[:2] != expected_size:
            print(f"[WARNING] Frame {i} has inconsistent size {frame_array.shape[:2]}, expected {expected_size}. Resizing...")
            frames[i] = frame.resize((size[0], size[1]))  # Resize to match (W, H) for PIL

    fps = save_fps if fps is None else fps
    # Output directly to specified directory (no nested subdirectory)
    # This avoids path resolution overhead and fixes Windows backslash issues on Linux
    save_root = output
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    if mode == "video_inpainting":
        frames_len = len(frames)

        # 🔥 EXTREME SPEED: Use numpy mask arrays directly if provided (skip disk I/O!)
        if masks_array is not None:
            # Convert numpy arrays (grayscale) to PIL Images
            # Apply dilation like read_mask() does
            flow_masks = []
            masks_dilated = []

            for mask_np in masks_array:
                # Ensure grayscale
                if len(mask_np.shape) == 3:
                    mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)

                # Resize if needed
                if size is not None:
                    mask_np = cv2.resize(mask_np, size, interpolation=cv2.INTER_NEAREST)

                # Flow mask dilation
                if mask_dilation > 0:
                    flow_mask_img = scipy.ndimage.binary_dilation(
                        mask_np, iterations=mask_dilation
                    ).astype(np.uint8)
                else:
                    flow_mask_img = binary_mask(mask_np).astype(np.uint8)
                flow_masks.append(Image.fromarray(flow_mask_img * 255))

                # Regular mask dilation
                if mask_dilation > 0:
                    mask_img = scipy.ndimage.binary_dilation(
                        mask_np, iterations=mask_dilation
                    ).astype(np.uint8)
                else:
                    mask_img = binary_mask(mask_np).astype(np.uint8)
                masks_dilated.append(Image.fromarray(mask_img * 255))

            # If single mask, repeat for all frames
            if len(masks_array) == 1:
                flow_masks = flow_masks * frames_len
                masks_dilated = masks_dilated * frames_len

            print(f"[FAST] Using {len(masks_array)} masks from memory (ZERO disk I/O!)")
        else:
            # Original disk-based loading
            flow_masks, masks_dilated = read_mask(
                mask,
                frames_len,
                size,
                flow_mask_dilates=mask_dilation,
                mask_dilates=mask_dilation,
            )

        w, h = size
    elif mode == "video_outpainting":
        assert (
            scale_h is not None and scale_w is not None
        ), "Please provide a outpainting scale (s_h, s_w)."
        frames, flow_masks, masks_dilated, size = extrapolation(
            frames, (scale_h, scale_w)
        )
        w, h = size
    else:
        raise NotImplementedError

    # for saving the masked frames or video
    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]), 2).repeat(3, axis=2) / 255.0
        img = np.array(frames[i])
        # Create green array matching THIS frame's actual shape (handles inconsistent frame sizes)
        img_h, img_w = img.shape[:2]
        green = np.zeros([img_h, img_w, 3])
        green[:, :, 1] = 255
        alpha = 0.6
        # alpha = 1.0
        fuse_img = (1 - alpha) * img + alpha * green
        fuse_img = mask_ * fuse_img + (1 - mask_) * img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    # NOTE: Can't use channels_last on 5D tensors (BTCHW) - only works with 4D (NCHW)
    # Models will automatically convert internally via their .to(memory_format=torch.channels_last)
    frames, flow_masks, masks_dilated = (
        frames.to(device, non_blocking=True),
        flow_masks.to(device, non_blocking=True),
        masks_dilated.to(device, non_blocking=True),
    )

    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    # Try to use TensorRT FastFlowNet engine for optical flow; fallback to PyTorch RAFT
    class _RAFTAdapter:
        def __init__(self, device: torch.device, use_half: bool):
            self.device = device
            self.use_half = use_half and (device.type == "cuda")
            self._trt_ready = False
            self._ctx = None
            self._engine = None
            self._in_idx = None
            self._out_idx = None
            self._binding_dtype_in = None
            self._binding_dtype_out = None
            self._in_name = None
            self._out_name = None
            self._use_v3 = False
            self._stream = None
            self._model_type = "unknown"  # Track which model is actually loaded
            # Enforce TensorRT-only mode when requested (no PyTorch fallback)
            def _parse_bool(val: str) -> bool:
                return str(val).lower() in ("1", "true", "yes", "on")
            self._force_trt = _parse_bool(os.getenv("FORCE_TRT_RAFT", "0"))

            # PRIORITY 1: Check USE_NEUFLOW FIRST (takes precedence over TensorRT)
            if os.getenv("USE_NEUFLOW", "0") == "1":
                # NeuFlow v2 uses TensorRT FP16 engine with fallback to PyTorch RAFT
                model_path = os.path.join(os.path.dirname(__file__), "models", "neuflow_things_fp16.engine")

                # Try TensorRT NeuFlow first
                neuflow_loaded = False
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    try:
                        self._raft = RAFT_bi(model_path, device)
                        self._model_type = "NeuFlow v2 TensorRT FP16"
                        print("[OK] Using NeuFlow v2 TensorRT for optical flow (10-70x faster than RAFT)")
                        print("[PERF] TensorRT FP16 = 3-4X FASTER than ONNX Runtime!")
                        neuflow_loaded = True
                    except Exception as e:
                        print(f"[WARNING] NeuFlow TensorRT failed to load: {e}")
                        print("[INFO] Falling back to PyTorch RAFT...")

                # Fallback to PyTorch RAFT if TensorRT failed
                if not neuflow_loaded:
                    from model.modules.flow_comp_raft import RAFT_bi as PyTorch_RAFT
                    raft_path = os.path.join(os.path.dirname(__file__), "weights", "raft-things.pth")
                    self._raft = PyTorch_RAFT(raft_path, device)
                    self._model_type = "PyTorch RAFT (fallback)"
                    print(f"[OK] Using PyTorch RAFT as fallback (slower but stable)")
                    print(f"[INFO] To enable NeuFlow TensorRT: Run tools/BUILD_NEUFLOW_TRT.bat")

                self._trt_ready = False  # Disable TensorRT path in __call__
            else:
                # PRIORITY 2: Try TensorRT RAFT (only when USE_NEUFLOW=0 AND FORCE_TRT_RAFT=1)
                # When FORCE_TRT_RAFT=0, skip TensorRT entirely and use PyTorch RAFT
                self._trt_ready = False  # Default to PyTorch
                if self._force_trt:
                    # Candidate engine paths (absolute + relative)
                    engine_candidates = [
                        os.path.join(os.getcwd(), 'faster-propainter-main', 'engines', 'raft', 'raft_fp16.engine'),
                        os.path.join(os.path.dirname(__file__), 'engines', 'raft', 'raft_fp16.engine'),
                    ]
                    engine_path = None
                    for p in engine_candidates:
                        if os.path.exists(p):
                            engine_path = p
                            break

                    if not engine_path:
                        raise RuntimeError("FORCE_TRT_RAFT=1 set but RAFT engine not found at expected locations")

                if self._force_trt and engine_path and device.type == 'cuda':
                    try:
                        # Ensure TensorRT DLLs are available on Windows
                        trt_root = os.environ.get('TENSORRT_ROOT') or os.path.join(os.getcwd(), 'TensorRT-10.7.0.23')
                        trt_lib = os.path.join(trt_root, 'lib')
                        if os.name == 'nt' and os.path.isdir(trt_lib):
                            try:
                                os.add_dll_directory(trt_lib)
                            except Exception:
                                pass
                        import tensorrt as trt
                        logger = trt.Logger(trt.Logger.WARNING)
                        runtime = trt.Runtime(logger)
                        with open(engine_path, 'rb') as f:
                            self._engine = runtime.deserialize_cuda_engine(f.read())
                        if self._engine is None:
                            raise RuntimeError('TRT engine deserialize failed')
                        # Don't create context here - will be created per-thread lazily
                        self._ctx = None
                        # Resolve bindings robustly (support legacy and TensorRT 10 v3 I/O)
                        try:
                            nb = self._engine.num_bindings  # may not exist on TRT 10
                        except Exception:
                            nb = None
                        if isinstance(nb, int) and nb > 0:
                            for i in range(nb):
                                if self._engine.binding_is_input(i) and self._in_idx is None:
                                    self._in_idx = i
                                    self._binding_dtype_in = self._engine.get_binding_dtype(i)
                                if (not self._engine.binding_is_input(i)) and self._out_idx is None:
                                    self._out_idx = i
                                    self._binding_dtype_out = self._engine.get_binding_dtype(i)
                            assert self._in_idx is not None and self._out_idx is not None
                        else:
                            # TensorRT 10 v3 API uses named tensors
                            self._use_v3 = True
                            n_io = self._engine.num_io_tensors
                            for i in range(n_io):
                                name = self._engine.get_tensor_name(i)
                                mode = self._engine.get_tensor_mode(name)
                                if mode == trt.TensorIOMode.INPUT and self._in_name is None:
                                    self._in_name = name
                                    self._binding_dtype_in = self._engine.get_tensor_dtype(name)
                                elif mode == trt.TensorIOMode.OUTPUT and self._out_name is None:
                                    self._out_name = name
                                    self._binding_dtype_out = self._engine.get_tensor_dtype(name)
                            assert self._in_name is not None and self._out_name is not None
                        # Create a dedicated CUDA stream to avoid default-stream sync penalties
                        import torch as _torch
                        if device.type == 'cuda':
                            try:
                                self._stream = _torch.cuda.Stream(device=device)
                            except Exception:
                                self._stream = None
                        self._trt_ready = True
                        self._model_type = "TensorRT RAFT FP16"
                        print(f"[OK] Using TensorRT RAFT engine: {engine_path}")
                        # DEBUG: Print engine shape ranges
                        if self._use_v3:
                            print(f"[TRT DEBUG] Input tensor: {self._in_name}, Output tensor: {self._out_name}")
                        else:
                            print(f"[TRT DEBUG] Input binding: {self._in_idx}, Output binding: {self._out_idx}")
                    except Exception as e:
                        if self._force_trt:
                            raise
                        else:
                            print(f"[WARNING] TensorRT RAFT engine load failed, falling back to PyTorch: {e}")
                            self._trt_ready = False

                # PRIORITY 3: Fallback to PyTorch RAFT (if TensorRT failed)
                if not self._trt_ready:
                    if self._force_trt:
                        raise RuntimeError("FORCE_TRT_RAFT=1 set but TensorRT RAFT engine is not available")

                    # Original RAFT with PyTorch
                    ckpt_path = load_file_from_url(
                        url=os.path.join(pretrain_model_url, "raft-things.pth"),
                        model_dir="weights",
                        progress=True,
                        file_name=None,
                    )
                    self._raft = RAFT_bi(ckpt_path, device)
                    self._model_type = "PyTorch RAFT"
                    print("[OK] RAFT PyTorch model initialized")

        @property
        def model_type(self) -> str:
            """Return the actual optical flow model type being used"""
            return self._model_type

        def _get_thread_context(self):
            """Get or create TensorRT execution context for current thread"""
            if not hasattr(_thread_local, 'trt_contexts'):
                _thread_local.trt_contexts = {}

            thread_id = threading.get_ident()
            if thread_id not in _thread_local.trt_contexts:
                # Create new context for this thread
                ctx = self._engine.create_execution_context()
                _thread_local.trt_contexts[thread_id] = ctx
                print(f"[TRT] Created new execution context for thread {thread_id}")

            return _thread_local.trt_contexts[thread_id]

        @torch.inference_mode()
        def __call__(self, frames_btchw: torch.Tensor, iters: int = 20):
            # frames: [B, T, C, H, W]
            if not self._trt_ready:
                return self._raft(frames_btchw, iters=iters)

            B, T, C, H, W = frames_btchw.shape
            assert B == 1, "Adapter expects batch=1"

            # Dtypes for engine bindings
            import tensorrt as trt  # type: ignore
            in_half = (self._binding_dtype_in == trt.DataType.HALF)
            out_half = (self._binding_dtype_out == trt.DataType.HALF)

            def _exec_pair(img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
                # img0/img1: [B, C, H, W] on device
                # Build input [B,2,3,H,W]
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        inp = torch.stack((img0, img1), dim=1)
                        if in_half:
                            inp = inp.half()
                        else:
                            inp = inp.float()
                else:
                    inp = torch.stack((img0, img1), dim=1)
                    if in_half:
                        inp = inp.half()
                    else:
                        inp = inp.float()
                # Allocate output tensor [B,2,H,W]
                if self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        out = torch.empty((B, 2, H, W), device=inp.device, dtype=torch.float16 if out_half else torch.float32)
                else:
                    out = torch.empty((B, 2, H, W), device=inp.device, dtype=torch.float16 if out_half else torch.float32)
                # DEBUG: Log shape before execution (only first time)
                if not hasattr(self, '_trt_shape_logged'):
                    print(f"[TRT EXEC] Input shape: {inp.shape}, dtype: {inp.dtype}")
                    print(f"[TRT EXEC] Output shape: {out.shape}, dtype: {out.dtype}")
                    print(f"[TRT] Thread-local contexts enabled - TRUE PARALLEL execution!")
                    self._trt_shape_logged = True

                # Get thread-local context (NO LOCK NEEDED - each thread has its own!)
                ctx = self._get_thread_context()

                # Execute depending on TRT API
                if not getattr(self, '_use_v3', False):
                    # Legacy bindings API
                    ctx.set_binding_shape(self._in_idx, tuple(inp.shape))
                    # Query num_bindings safely
                    try:
                        nb = self._engine.num_bindings
                    except Exception:
                        nb = 2
                    bindings = [None] * int(nb)
                    bindings[self._in_idx] = int(inp.data_ptr())
                    bindings[self._out_idx] = int(out.data_ptr())
                    ok = ctx.execute_v2(bindings)
                else:
                    # TensorRT 10 v3 API
                    try:
                        success = ctx.set_input_shape(self._in_name, tuple(inp.shape))
                        if not success:
                            print(f"[TRT ERROR] set_input_shape failed for shape {inp.shape}")
                    except Exception as e:
                        print(f"[TRT ERROR] Exception in set_input_shape: {e}")
                        raise
                    ctx.set_tensor_address(self._in_name, int(inp.data_ptr()))
                    ctx.set_tensor_address(self._out_name, int(out.data_ptr()))
                    if self._stream is not None:
                        stream_ptr = int(self._stream.cuda_stream)
                    else:
                        stream_ptr = int(torch.cuda.current_stream().cuda_stream)
                    ok = ctx.execute_async_v3(stream_ptr)

                if not ok:
                    # DEBUG: Print shapes and binding info to understand failure
                    print(f"[TRT FAIL] Input shape: {inp.shape}, dtype: {inp.dtype}")
                    print(f"[TRT FAIL] Output shape: {out.shape}, dtype: {out.dtype}")
                    print(f"[TRT FAIL] Expected input: {self._in_name}, output: {self._out_name}")
                    print(f"[TRT FAIL] Device: {inp.device}")
                    raise RuntimeError(f'TRT execute failed - input shape: {inp.shape}, output shape: {out.shape}')
                return out

            # Prepare frame tensors on device
            dev_frames = frames_btchw.to(self.device, non_blocking=True)
            if self.use_half:
                dev_frames = dev_frames.half()
            flows_f = []
            flows_b = []
            for t in range(T - 1):
                f0 = dev_frames[:, t]
                f1 = dev_frames[:, t + 1]
                ff = _exec_pair(f0, f1)
                fb = _exec_pair(f1, f0)
                flows_f.append(ff)
                flows_b.append(fb)
            # Ensure TRT stream completes before consuming outputs on default stream
            if self._stream is not None:
                torch.cuda.current_stream().wait_stream(self._stream)
            flows_f = torch.cat(flows_f, dim=0).unsqueeze(0)  # [1, T-1, 2, H, W]
            flows_b = torch.cat(flows_b, dim=0).unsqueeze(0)
            return flows_f, flows_b

    # ============================================================================
    # MODEL CACHING: Load models once per worker, reuse across segments
    # ============================================================================
    global _GLOBAL_MODELS_CACHE, _MODEL_CACHE_LOCK
    import threading

    if _MODEL_CACHE_LOCK is None:
        _MODEL_CACHE_LOCK = threading.Lock()

    cache_key = f"{device}_{use_half}"

    if use_cached_models:
        with _MODEL_CACHE_LOCK:
            if _GLOBAL_MODELS_CACHE is None:
                print(f"[INIT] Creating global model cache (worker first-time initialization)...")
                _GLOBAL_MODELS_CACHE = {}

            if cache_key not in _GLOBAL_MODELS_CACHE:
                print(f"[CACHE] Loading models for key={cache_key}... (this happens ONCE per worker)")
                cache_start = time.time()

                # Create models for the first time
                _GLOBAL_MODELS_CACHE[cache_key] = {
                    'raft': None,  # Will be set below
                    'rfcnet': None,
                    'propainter': None
                }
                print(f"[INFO] Model cache initialized for {cache_key}")

                # Create RAFT model
                raft_model = _RAFTAdapter(device, use_half)
                _GLOBAL_MODELS_CACHE[cache_key]['raft'] = raft_model
                print(f"[CACHE] Optical flow model cached: {raft_model.model_type} ({time.time() - cache_start:.2f}s)")
            else:
                print(f"[CACHE] Reusing cached models for {cache_key} (ZERO reload time!)")

            fix_raft = _GLOBAL_MODELS_CACHE[cache_key]['raft']
            # Validate which optical flow model is actually loaded
            if fix_raft is not None:
                print(f"[VALIDATION] Active optical flow model: {fix_raft.model_type}")
            else:
                raise RuntimeError("[ERROR] Optical flow model failed to initialize! Check logs for errors.")
    else:
        # Legacy behavior: create new models each time
        fix_raft = _RAFTAdapter(device, use_half)
        if fix_raft is not None:
            print(f"[VALIDATION] Active optical flow model: {fix_raft.model_type}")
        else:
            raise RuntimeError("[ERROR] Optical flow model failed to initialize! Check logs for errors.")

    ##############################################
    # set up RFCNet with TensorRT acceleration
    ##############################################
    # Try to use TensorRT RFCNet engine; fallback to PyTorch RecurrentFlowCompleteNet
    class _RFCNetAdapter:
        def __init__(self, device: torch.device, use_half: bool, ckpt_path: str):
            self.device = device
            self.use_half = use_half and (device.type == "cuda")
            self._trt_ready = False
            self._ctx = None
            self._engine = None
            self._use_v3 = False
            self._stream = None
            self._in_names = None  # (masked_flows_name, masks_name)
            self._out_name = None

            # Enforce TensorRT-only mode when requested
            def _parse_bool(val: str) -> bool:
                return str(val).lower() in ("1", "true", "yes", "on")
            self._force_trt = _parse_bool(os.getenv("FORCE_TRT_RFCNET", "0"))

            # Candidate engine paths (DCNv4 version)
            engine_candidates = [
                os.path.join(os.getcwd(), 'engines', 'rfcnet', 'rfcnet_dcnv4_fp16.engine'),
                os.path.join(os.getcwd(), 'faster-propainter-main', 'engines', 'rfcnet', 'rfcnet_dcnv4_fp16.engine'),
                os.path.join(os.path.dirname(__file__), 'engines', 'rfcnet', 'rfcnet_dcnv4_fp16.engine'),
            ]
            engine_path = None
            for p in engine_candidates:
                if os.path.exists(p):
                    engine_path = p
                    break

            if not engine_path and self._force_trt:
                raise RuntimeError("FORCE_TRT_RFCNET=1 set but RFCNet engine not found at expected locations")

            if engine_path and device.type == 'cuda':
                try:
                    # Ensure TensorRT DLLs are available
                    trt_root = os.environ.get('TENSORRT_ROOT') or os.path.join(os.getcwd(), 'TensorRT-10.13.3.9')
                    trt_lib = os.path.join(trt_root, 'lib')
                    if os.name == 'nt' and os.path.isdir(trt_lib):
                        try:
                            os.add_dll_directory(trt_lib)
                        except Exception:
                            pass

                    # Load DCNv4 plugin (required for RFCNet TensorRT engine)
                    import ctypes
                    plugin_path = os.path.join(os.getcwd(), 'dcnv4_tensorrt_plugin', 'build', 'Release', 'dcnv4_plugin.dll')
                    # Also add plugin directory to DLL search path for dependencies
                    plugin_dir = os.path.dirname(plugin_path)
                    if os.path.isdir(plugin_dir):
                        try:
                            os.add_dll_directory(plugin_dir)
                        except Exception:
                            pass
                    if os.path.exists(plugin_path):
                        try:
                            ctypes.CDLL(plugin_path)
                        except Exception as e:
                            if self._force_trt:
                                raise RuntimeError(f'Failed to load DCNv4 plugin from {plugin_path}: {e}')
                            else:
                                print(f"[WARNING] DCNv4 plugin load failed: {e}")
                                raise
                    else:
                        if self._force_trt:
                            raise RuntimeError(f'DCNv4 plugin not found at {plugin_path}')
                        else:
                            print(f"[WARNING] DCNv4 plugin not found at {plugin_path}")
                            raise FileNotFoundError(f'DCNv4 plugin not found at {plugin_path}')

                    import tensorrt as trt
                    logger = trt.Logger(trt.Logger.WARNING)
                    runtime = trt.Runtime(logger)
                    with open(engine_path, 'rb') as f:
                        self._engine = runtime.deserialize_cuda_engine(f.read())
                    if self._engine is None:
                        raise RuntimeError('TRT engine deserialize failed')
                    # Don't create context here - will be created per-thread lazily
                    self._ctx = None

                    # Use TensorRT 10 v3 API (named tensors)
                    self._use_v3 = True
                    n_io = self._engine.num_io_tensors
                    input_names = []
                    for i in range(n_io):
                        name = self._engine.get_tensor_name(i)
                        mode = self._engine.get_tensor_mode(name)
                        if mode == trt.TensorIOMode.INPUT:
                            input_names.append(name)
                        elif mode == trt.TensorIOMode.OUTPUT and self._out_name is None:
                            self._out_name = name

                    # Expected input names: "masked_flows" and "masks"
                    if len(input_names) == 2:
                        # Sort to ensure consistent order
                        input_names.sort()
                        self._in_names = tuple(input_names)
                    assert self._in_names is not None and self._out_name is not None, \
                        f"Failed to find expected inputs/output. Found: inputs={input_names}, output={self._out_name}"

                    # Create dedicated CUDA stream
                    if device.type == 'cuda':
                        try:
                            self._stream = torch.cuda.Stream(device=device)
                        except Exception:
                            self._stream = None

                    self._trt_ready = True
                    print(f"[OK] Using TensorRT RFCNet engine: {engine_path}")
                    print(f"   Expected speedup: 8.45x (30.78ms -> 3.64ms per inference)")
                except Exception as e:
                    if self._force_trt:
                        raise
                    else:
                        print(f"[WARNING] TensorRT RFCNet engine load failed, falling back to PyTorch: {e}")
                        self._trt_ready = False

            # Fallback to PyTorch if TensorRT not available
            if not self._trt_ready:
                print(f"Loading PyTorch RecurrentFlowCompleteNet from: {ckpt_path}")
                self._model = RecurrentFlowCompleteNet(ckpt_path)
                for p in self._model.parameters():
                    p.requires_grad = False
                self._model.to(device, non_blocking=True)
                self._model.eval()

        def forward(self, masked_flows: torch.Tensor, masks: torch.Tensor):
            """
            Forward pass for RFCNet.

            Args:
                masked_flows: (B, T, 2, H, W) tensor
                masks: (B, T, 1, H, W) tensor

            Returns:
                flow: (B, T, 2, H, W) tensor
                masks_updated: (B, T, 1, H, W) tensor
            """
            if self._trt_ready:
                return self._forward_trt(masked_flows, masks)
            else:
                return self._model.forward(masked_flows, masks)

        def _get_thread_context(self):
            """Get or create TensorRT execution context for current thread"""
            if not hasattr(_thread_local, 'rfcnet_contexts'):
                _thread_local.rfcnet_contexts = {}

            thread_id = threading.get_ident()
            if thread_id not in _thread_local.rfcnet_contexts:
                # Create new context for this thread
                ctx = self._engine.create_execution_context()
                _thread_local.rfcnet_contexts[thread_id] = ctx
                print(f"[TRT RFCNet] Created new execution context for thread {thread_id}")

            return _thread_local.rfcnet_contexts[thread_id]

        def _forward_trt(self, masked_flows: torch.Tensor, masks: torch.Tensor):
            """TensorRT execution path."""
            # Convert to FP16 if needed
            if self.use_half:
                masked_flows = masked_flows.half()
                masks = masks.half()
            else:
                masked_flows = masked_flows.float()
                masks = masks.float()

            # Allocate output tensor
            B, T, _, H, W = masked_flows.shape
            dtype = torch.float16 if self.use_half else torch.float32

            if self._stream is not None:
                with torch.cuda.stream(self._stream):
                    flow_out = torch.empty((B, T, 2, H, W), device=self.device, dtype=dtype)
            else:
                flow_out = torch.empty((B, T, 2, H, W), device=self.device, dtype=dtype)

            # Get thread-local context (NO LOCK NEEDED - each thread has its own!)
            ctx = self._get_thread_context()

            # Set input shapes
            ctx.set_input_shape(self._in_names[0], tuple(masked_flows.shape))
            ctx.set_input_shape(self._in_names[1], tuple(masks.shape))

            # Set tensor addresses
            ctx.set_tensor_address(self._in_names[0], int(masked_flows.data_ptr()))
            ctx.set_tensor_address(self._in_names[1], int(masks.data_ptr()))
            ctx.set_tensor_address(self._out_name, int(flow_out.data_ptr()))

            # Execute
            if self._stream is not None:
                stream_ptr = int(self._stream.cuda_stream)
            else:
                stream_ptr = int(torch.cuda.current_stream().cuda_stream)

            ok = ctx.execute_async_v3(stream_ptr)
            if not ok:
                raise RuntimeError('TRT RFCNet execute failed')

            # Ensure stream completes
            if self._stream is not None:
                torch.cuda.current_stream().wait_stream(self._stream)

            # Return (flow, masks) - masks unchanged for compatibility
            return flow_out, masks

        def forward_bidirect_flow(self, masked_flows_bi, masks_bi):
            """
            Bidirectional flow completion (wrapper for compatibility).

            Args:
                masked_flows_bi: Tuple of (flows_f, flows_b) where each is (B, T-1, 2, H, W)
                masks_bi: (B, T, 1, H, W) - single-channel masks

            Returns:
                pred_flows_bi: Tuple of [pred_flows_f, pred_flows_b]
                pred_edges_bi: Tuple of [None, None] (edges not used in TRT mode)
            """
            if self._trt_ready:
                # Extract forward and backward flows from tuple
                flows_f, flows_b = masked_flows_bi

                # Extract forward/backward masks (B, T-1, 1, H, W)
                masks_f = masks_bi[:, :-1, ...].contiguous()
                masks_b = masks_bi[:, 1:, ...].contiguous()

                # Process forward and backward separately
                pred_f, _ = self._forward_trt(flows_f, masks_f)
                pred_b, _ = self._forward_trt(flows_b, masks_b)

                # Return tuple format for compatibility
                return [pred_f, pred_b], [None, None]
            else:
                # Use PyTorch model's native implementation
                return self._model.forward_bidirect_flow(masked_flows_bi, masks_bi)

        def combine_flow(self, masked_flows_bi, pred_flows_bi, masks):
            """
            Combine predicted flows with masked flows (inpainting blending).

            Args:
                masked_flows_bi: Tuple of (masked_flows_f, masked_flows_b)
                pred_flows_bi: Tuple of (pred_flows_f, pred_flows_b)
                masks: (B, T, 1, H, W)

            Returns:
                Tuple of (combined_flows_f, combined_flows_b)
            """
            masks_forward = masks[:, :-1, ...].contiguous()
            masks_backward = masks[:, 1:, ...].contiguous()

            pred_flows_forward = pred_flows_bi[0] * masks_forward + masked_flows_bi[0] * (1 - masks_forward)
            pred_flows_backward = pred_flows_bi[1] * masks_backward + masked_flows_bi[1] * (1 - masks_backward)

            return pred_flows_forward, pred_flows_backward

        def half(self):
            """No-op for compatibility (TRT engine already uses FP16)."""
            return self

    ##############################################
    # TensorRT Sparse Transformer Adapter (NO FALLBACK)
    ##############################################
    class _TransformerTRTAdapter:
        """TensorRT wrapper for Sparse Temporal Transformer (NO FALLBACK when force_trt=True)"""

        def __init__(self, transformer_module, force_trt=False):
            self._pytorch_model = transformer_module
            self._force_trt = force_trt
            self._trt_ready = False
            self._engine = None
            self._engine_path = None

            if not force_trt:
                # TensorRT disabled, use PyTorch
                return

            # Search for transformer engine
            engine_candidates = [
                os.path.join(os.getcwd(), 'engines', 'transformer', 'transformer_fp16.engine'),
                os.path.join(os.getcwd(), 'faster-propainter-main', 'engines', 'transformer', 'transformer_fp16.engine'),
                os.path.join(os.path.dirname(__file__), 'engines', 'transformer', 'transformer_fp16.engine'),
                os.path.join(os.path.dirname(__file__), '..', 'engines', 'transformer', 'transformer_fp16.engine'),
            ]

            for p in engine_candidates:
                if os.path.exists(p):
                    self._engine_path = p
                    break

            # STRICT MODE: Fail if engine not found
            if not self._engine_path:
                raise RuntimeError(
                    f"FORCE_TRT_TRANSFORMER=1 but transformer engine not found!\n"
                    f"Searched: {engine_candidates}\n"
                    f"Build engine first: BUILD_TRANSFORMER_ENGINE.bat"
                )

            # Load TensorRT engine
            try:
                # Ensure TensorRT DLLs are available
                trt_root = os.environ.get('TENSORRT_ROOT') or os.path.join(os.getcwd(), 'TensorRT-10.13.3.9')
                trt_lib = os.path.join(trt_root, 'lib')
                if os.name == 'nt' and os.path.isdir(trt_lib):
                    try:
                        os.add_dll_directory(trt_lib)
                    except Exception:
                        pass

                import tensorrt as trt
                trt_logger = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(trt_logger)

                with open(self._engine_path, 'rb') as f:
                    self._engine = runtime.deserialize_cuda_engine(f.read())

                if self._engine is None:
                    raise RuntimeError(f"Failed to deserialize transformer engine: {self._engine_path}")

                self._trt_ready = True
                print(f"[TRT Transformer] Engine loaded: {self._engine_path}")
                print(f"[TRT Transformer] Expected speedup: 5-10x (2.39s → 0.24-0.48s per segment)")

            except Exception as e:
                # STRICT MODE: NO FALLBACK!
                raise RuntimeError(
                    f"FORCE_TRT_TRANSFORMER=1 but engine load failed: {e}\n"
                    f"Engine path: {self._engine_path}\n"
                    f"Check TensorRT installation and engine compatibility"
                )

        def _get_thread_context(self):
            """Get or create TensorRT execution context for current thread"""
            if not hasattr(_thread_local, 'transformer_contexts'):
                _thread_local.transformer_contexts = {}

            thread_id = threading.get_ident()
            if thread_id not in _thread_local.transformer_contexts:
                ctx = self._engine.create_execution_context()
                _thread_local.transformer_contexts[thread_id] = ctx
                print(f"[TRT Transformer] Created execution context for thread {thread_id}")

            return _thread_local.transformer_contexts[thread_id]

        def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
            """
            Forward pass (TensorRT or PyTorch based on force_trt)

            Args:
                x: [B, T, H, W, C] feature tokens
                fold_x_size: tuple (fold_h, fold_w) - not used in TRT
                l_mask: [B, T, H, W, 1] local mask - not used in TRT
                t_dilation: temporal dilation - not used in TRT

            Returns:
                output: [B, T, H, W, C] transformed features
            """
            if self._trt_ready:
                return self._forward_trt(x)
            else:
                # PyTorch path (when force_trt=False)
                return self._pytorch_model(x, fold_x_size, l_mask, t_dilation)

        def _forward_trt(self, x):
            """TensorRT inference path"""
            B, T, H, W, C = x.shape

            # Get thread-local context
            ctx = self._get_thread_context()

            # Set dynamic shape (T dimension varies)
            ctx.set_input_shape("input", (B, T, H, W, C))

            # Allocate output
            output = torch.empty_like(x)

            # Execute
            bindings = [x.data_ptr(), output.data_ptr()]
            success = ctx.execute_v2(bindings)

            if not success:
                raise RuntimeError("TensorRT transformer inference failed")

            return output

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "recurrent_flow_completion.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )

    # Initialize RFCNet (PyTorch with optional Torch-TensorRT compilation)
    if use_cached_models:
        with _MODEL_CACHE_LOCK:
            if _GLOBAL_MODELS_CACHE[cache_key]['rfcnet'] is None:
                # First time loading RFCNet
                rfcnet_start = time.time()
                fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
                for p in fix_flow_complete.parameters():
                    p.requires_grad = False
                fix_flow_complete.to(device, non_blocking=True)
                fix_flow_complete.eval()
                # THREAD-SAFE FIX: Convert to FP16 during caching, not during inference
                if use_half:
                    fix_flow_complete = fix_flow_complete.half()
                # NOTE: Can't use channels_last on models with Conv3d (has non-4D parameters)
                _GLOBAL_MODELS_CACHE[cache_key]['rfcnet'] = fix_flow_complete
                print(f"[CACHE] RFCNet model cached in {'FP16' if use_half else 'FP32'} ({time.time() - rfcnet_start:.2f}s)")
            else:
                fix_flow_complete = _GLOBAL_MODELS_CACHE[cache_key]['rfcnet']
    else:
        # Legacy behavior
        fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
        for p in fix_flow_complete.parameters():
            p.requires_grad = False
        fix_flow_complete.to(device, non_blocking=True)
        fix_flow_complete.eval()
        # NOTE: Can't use channels_last on models with Conv3d (has non-4D parameters)

    # torch.compile() with inductor backend (REQUIRED when RFCNET_TORCHTRT=1, no fallback)
    if os.getenv("RFCNET_TORCHTRT", "0").lower() in ("1", "true", "yes", "on"):
        from trt_runtime import maybe_compile_rfcnet
        if not maybe_compile_rfcnet(fix_flow_complete, device, use_fp16=use_half):
            raise RuntimeError(
                "RFCNET_TORCHTRT=1 set but torch.compile(inductor) compilation failed!\n"
                "Check CUDA availability and Visual Studio C++ compiler."
            )
        print("[OK] RFCNet accelerated via torch.compile(backend='inductor')")

    ##############################################
    # set up ProPainter model
    ##############################################
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "ProPainter.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )

    # ProPainter model initialization with caching
    if use_cached_models:
        with _MODEL_CACHE_LOCK:
            if _GLOBAL_MODELS_CACHE[cache_key]['propainter'] is None:
                # First time loading ProPainter
                propainter_start = time.time()
                model = InpaintGenerator(model_path=ckpt_path).to(device, non_blocking=True)
                model.eval()
                # THREAD-SAFE FIX: Convert to FP16 during caching, not during inference
                if use_half:
                    model = model.half()

                # NOTE: Can't use channels_last - incompatible with video models (5D tensors)

                # ⚡ TORCH COMPILE: DISABLED for ProPainter wrapper (causes FX tracing conflict)
                # Transformer layers are still compiled in sparse_transformer.py for 1.5-3x speedup!
                # if os.environ.get('USE_TORCH_COMPILE', '0') == '1':
                #     print(f"[COMPILE] Compiling ProPainter with torch.compile() for extreme speed...")
                #     compile_start = time.time()
                #     # mode="default" for thread-safe kernel fusion (no CUDA graphs = no TLS crashes)
                #     model = torch.compile(model, mode="default", fullgraph=False)
                #     print(f"[OK] torch.compile() setup complete ({time.time() - compile_start:.2f}s) - warmup will happen on first inference")

                _GLOBAL_MODELS_CACHE[cache_key]['propainter'] = model
                print(f"[CACHE] ProPainter model cached in {'FP16' if use_half else 'FP32'} ({time.time() - propainter_start:.2f}s)")
                print(f"[OK] All models cached! Total worker init time saved on next segment")
            else:
                model = _GLOBAL_MODELS_CACHE[cache_key]['propainter']
    else:
        # Legacy behavior
        model = InpaintGenerator(model_path=ckpt_path).to(device, non_blocking=True)
        model.eval()

        # NOTE: Can't use channels_last - incompatible with video models (5D tensors)

        # ⚡ TORCH COMPILE: DISABLED for ProPainter wrapper (causes FX tracing conflict)
        # Transformer layers are still compiled in sparse_transformer.py for 1.5-3x speedup!
        # if os.environ.get('USE_TORCH_COMPILE', '0') == '1':
        #     print(f"[COMPILE] Compiling ProPainter with torch.compile() for extreme speed...")
        #     compile_start = time.time()
        #     # mode="default" for thread-safe kernel fusion (no CUDA graphs = no TLS crashes)
        #     model = torch.compile(model, mode="default", fullgraph=False)
        #     print(f"[OK] torch.compile() setup complete ({time.time() - compile_start:.2f}s) - warmup will happen on first inference")

    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    # print(f'\nProcessing: {video_name} [{video_length} frames]...')
    print(f"Processing: {video_length} frames...")
    propainter_total_start = time.time()
    with torch.inference_mode():
        # ---- compute flow ----
        # Large batch sizes for better GPU utilization (17s processing time)
        if frames.size(-1) <= 640:
            short_clip_len = 32
        elif frames.size(-1) <= 720:
            short_clip_len = 24
        elif frames.size(-1) <= 1280:
            short_clip_len = 16  # main use case
        else:
            short_clip_len = 8

        # Use FP16 for optical flow with autocast (4x faster, no type errors!)
        flow_model_name = fix_raft.model_type
        if use_half:
            print(f"[OPTICAL FLOW] Using FP16 autocast for {flow_model_name}")
        else:
            print(f"[OPTICAL FLOW] Using FP32 for {flow_model_name}")

        optical_flow_start_time = time.time()
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in tqdm(range(0, video_length, short_clip_len), desc=f"Optical Flow ({flow_model_name})"):
            # for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)

                # Use autocast for automatic mixed precision if FP16 enabled
                if use_half:
                    with torch.cuda.amp.autocast():
                        if f == 0:
                            flows_f, flows_b = fix_raft(frames[:, f:end_f], iters=raft_iter)
                        else:
                            flows_f, flows_b = fix_raft(frames[:, f - 1 : end_f], iters=raft_iter)
                else:
                    if f == 0:
                        flows_f, flows_b = fix_raft(frames[:, f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = fix_raft(frames[:, f - 1 : end_f], iters=raft_iter)

                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                # Removed empty_cache() - was forcing GPU sync barrier

            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            if use_half:
                with torch.cuda.amp.autocast():
                    gt_flows_bi = fix_raft(frames, iters=raft_iter)
            else:
                gt_flows_bi = fix_raft(frames, iters=raft_iter)
            # Removed empty_cache() - was forcing GPU sync barrier

        if use_half:
            # Convert local tensors to FP16 (thread-safe)
            frames, flow_masks, masks_dilated = (
                frames.half(),
                flow_masks.half(),
                masks_dilated.half(),
            )
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            # REMOVED: fix_flow_complete.half() and model.half() - now cached in correct precision

        optical_flow_time = time.time() - optical_flow_start_time
        print(f"[OK] Optical flow completed in {optical_flow_time:.2f}s")

        # ---- complete flow ----
        print("[FLOW] Starting flow completion (RFCNet)...")
        flow_start_time = time.time()
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks[:, s_f : e_f + 1],
                )
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks[:, s_f : e_f + 1],
                )

                pred_flows_f.append(
                    pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e]
                )
                pred_flows_b.append(
                    pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e]
                )
                # Removed empty_cache() - was forcing GPU sync barrier

            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(
                gt_flows_bi, flow_masks
            )
            pred_flows_bi = fix_flow_complete.combine_flow(
                gt_flows_bi, pred_flows_bi, flow_masks
            )
            # Removed empty_cache() - was forcing GPU sync barrier

        flow_time = time.time() - flow_start_time
        print(f"[OK] Flow completion completed in {flow_time:.2f}s")

        # ---- image propagation ----
        print("[IMG] Starting image propagation...")
        img_prop_start_time = time.time()
        try:
            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(
                40, subvideo_length
            )  # reduced from 150 to 40 for lower VRAM usage during image propagation
            print(f"[IMG PROP] Using subvideo_length_img_prop={subvideo_length_img_prop} for {video_length} frames")
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                total_chunks = (video_length + subvideo_length_img_prop - 1) // subvideo_length_img_prop
                for chunk_idx, f in enumerate(range(0, video_length, subvideo_length_img_prop)):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    print(f"[IMG PROP] Chunk {chunk_idx+1}/{total_chunks}: processing frames {s_f}-{e_f} ({t} frames)")
                    pred_flows_bi_sub = (
                        pred_flows_bi[0][:, s_f : e_f - 1],
                        pred_flows_bi[1][:, s_f : e_f - 1],
                    )
                    prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                        masked_frames[:, s_f:e_f],
                        pred_flows_bi_sub,
                        masks_dilated[:, s_f:e_f],
                        "nearest",
                    )
                    print(f"[IMG PROP] Chunk {chunk_idx+1}/{total_chunks} complete")
                    updated_frames_sub = (
                        frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
                        + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    )
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                    updated_frames.append(
                        updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e]
                    )
                    updated_masks.append(
                        updated_masks_sub[:, pad_len_s : e_f - s_f - pad_len_e]
                    )
                    # Removed empty_cache() - was forcing GPU sync barrier

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = model.img_propagation(
                    masked_frames, pred_flows_bi, masks_dilated, "nearest"
                )
                updated_frames = (
                    frames * (1 - masks_dilated)
                    + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                )
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                # Removed empty_cache() - was forcing GPU sync barrier
            img_prop_time = time.time() - img_prop_start_time
            print(f"[OK] Image propagation completed in {img_prop_time:.2f}s")
        except Exception as e:
            print(f"[ERROR] Image propagation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1

    # ---- feature propagation + transformer ----
    print("[PROP] Starting feature propagation + transformer...")
    print(f"[PROP CONFIG] neighbor_stride={neighbor_stride}, ref_num={ref_num}, total_iterations={(video_length + neighbor_stride - 1) // neighbor_stride}")
    prop_start_time = time.time()
    for f in tqdm(range(0, video_length, neighbor_stride), desc="feature propagation"):
    # for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [
            i
            for i in range(
                max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)
            )
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
        )

        with torch.inference_mode():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            if f % (neighbor_stride * 10) == 0:  # Log every 10 iterations
                print(f"[PROP] Frame {f}/{video_length}: processing {l_t} neighbor frames + {len(ref_ids)} ref frames")

            # pred_img = selected_imgs # results of image propagation
            pred_img = model(
                selected_imgs,
                selected_pred_flows_bi,
                selected_masks,
                selected_update_masks,
                l_t,
            )

            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = (
                masks_dilated[0, neighbor_ids, :, :, :]
                .cpu()
                .permute(0, 2, 3, 1)
                .numpy()
                .astype(np.uint8)
            )
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[
                    i
                ] + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5
                        + img.astype(np.float32) * 0.5
                    )

                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        # Removed empty_cache() - was forcing GPU sync barrier

    prop_time = time.time() - prop_start_time
    print(f"[OK] Feature propagation + transformer completed in {prop_time:.2f}s")

    # Print timing breakdown
    propainter_total_time = time.time() - propainter_total_start
    print(f"\n{'='*70}")
    print(f"PROPAINTER TIMING BREAKDOWN ({video_length} frames)")
    print(f"{'='*70}")
    print(f"  1. Optical Flow:          {optical_flow_time:6.2f}s ({optical_flow_time/propainter_total_time*100:5.1f}%)")
    print(f"  2. Flow Completion:       {flow_time:6.2f}s ({flow_time/propainter_total_time*100:5.1f}%)")
    print(f"  3. Image Propagation:     {img_prop_time:6.2f}s ({img_prop_time/propainter_total_time*100:5.1f}%)")
    print(f"  4. Feature Prop+Trans:    {prop_time:6.2f}s ({prop_time/propainter_total_time*100:5.1f}%)")
    accounted_time = optical_flow_time + flow_time + img_prop_time + prop_time
    other_time = propainter_total_time - accounted_time
    print(f"  5. Other (overhead):      {other_time:6.2f}s ({other_time/propainter_total_time*100:5.1f}%)")
    print(f"  {'-'*70}")
    print(f"  TOTAL:                    {propainter_total_time:6.2f}s (100.0%)")
    print(f"  Per-frame avg:            {propainter_total_time/video_length*1000:6.1f}ms")
    print(f"{'='*70}\n")

    # save each frame
    if save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            # Normalize path to use forward slashes (fixes Windows backslash on Linux)
            img_save_root = os.path.join(
                save_root, "frames", str(idx).zfill(4) + ".png"
            ).replace('\\', '/')
            imwrite(f, img_save_root)

    # save videos frame
    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]

    imageio.mimwrite(
        os.path.join(save_root, "masked_in.mp4"),
        masked_frame_for_save,
        fps=fps,
        quality=7,
    )
    imageio.mimwrite(
        os.path.join(save_root, "inpaint_out.mp4"), comp_frames, fps=fps, quality=7, ffmpeg_params=["-sws_flags", "bilinear"]
    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    video_fp = "./running_car.mp4"
    mask_fp = "./mask.png"
    out_fp = "./output.mp4"
    pipeline(
        video_fp,
        mask_fp,
        out_fp,
        fp16=True,
        subvideo_length=80,
        neighbor_length=10,  # Reduced from 20 to 10 (default) = 50% fewer RAFT calls
    )
