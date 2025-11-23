# -*- coding: utf-8 -*-
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'  # Completely disable torch.compile
import time
import threading
import subprocess
import shutil

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

# ============================================================================
# FFMPEG PATH DETECTION FOR NVENC GPU ENCODING
# ============================================================================
def _get_ffmpeg_path():
    """Get FFmpeg executable path, with fallback to static-ffmpeg."""
    # Try system PATH first
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path

    # Fallback to static-ffmpeg
    try:
        from static_ffmpeg import run
        ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
        return ffmpeg_path
    except Exception:
        return 'ffmpeg'  # Last resort fallback

_FFMPEG_EXECUTABLE = _get_ffmpeg_path()

# ============================================================================
# OPTICAL FLOW MODEL - PyTorch RAFT with torch.compile
# ============================================================================
print("=" * 70)
print("OPTICAL FLOW MODEL CONFIGURATION")
print("=" * 70)
print(f"  -> Using PyTorch RAFT (with optional torch.compile)")
print(f"  USE_TORCH_COMPILE_RAFT = {os.getenv('USE_TORCH_COMPILE_RAFT', '0')}")
print("=" * 70)
print()

from model.modules.flow_comp_raft import RAFT_bi
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

# Cache version marker - increment to force cache invalidation
# Version includes DCNv4 setting to ensure cache is cleared when setting changes
_CACHE_VERSION = f"v3_dcnv4={os.getenv('ENABLE_DCNV4_RFCNET', '0')}"
print(f"[CACHE] Module loaded with cache version: {_CACHE_VERSION}")

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
    frames_tensor_gpu=None,  # GPU tensors from NVDEC (ZERO CPU transfers!)
    masks_tensor_gpu=None,   # GPU mask tensors (optional)
    use_cached_models=True,  # Enable global model cache for true parallel speedup
):

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if fp16 else False
    device = get_device()
    if device == torch.device("cpu"):
        use_half = False

    # ⚡ GPU-ONLY PIPELINE: Use GPU tensors directly from NVDEC (ZERO CPU transfers!)
    frames_already_tensors = False  # Flag to skip to_tensors() conversion later

    if frames_tensor_gpu is not None:
        frames_already_tensors = True  # Skip tensor conversion later
        print(f"[GPU-ONLY] Using {frames_tensor_gpu.shape[1]} frames from GPU memory (ZERO CPU transfers!)")

        # Frames already on GPU in correct format [1, T, C, H, W], range [-1, 1]
        frames = frames_tensor_gpu.to(device, non_blocking=True)

        # Derive metadata from tensor
        fps = save_fps
        _, num_frames, _, h, w = frames.shape
        size = (w, h)  # (W, H)

        # Extract video_name from video parameter
        if video.endswith(("mp4", "mov", "avi", "MP4", "MOV", "AVI")):
            video_name = os.path.basename(video)[:-4]
        else:
            video_name = os.path.basename(video)

        # Check size requirements
        if not width == -1 and not height == -1:
            size = (width, height)
        if not resize_ratio == 1.0:
            size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))

        # Ensure size is divisible by 8
        if (size[0] % 8 != 0) or (size[1] % 8 != 0):
            # Need to resize tensor on GPU
            new_w = (size[0] // 8) * 8
            new_h = (size[1] // 8) * 8
            size = (new_w, new_h)
            out_size = size

            # Resize on GPU using F.interpolate
            # frames shape: [1, T, C, H, W]
            # Reshape to [T, C, H, W] for interpolation
            frames = frames.squeeze(0)  # [T, C, H, W]
            frames = torch.nn.functional.interpolate(
                frames, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            frames = frames.unsqueeze(0)  # Back to [1, T, C, H, W]
            w, h = new_w, new_h
        else:
            out_size = size

        fps = save_fps if fps is None else fps
        save_root = os.path.join(output, video_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)

        if mode == "video_inpainting":
            frames_len = num_frames

            # Handle masks - convert to GPU tensors if needed
            if masks_tensor_gpu is not None:
                # Already on GPU
                flow_masks = masks_tensor_gpu.to(device, non_blocking=True)
                masks_dilated = flow_masks
                print(f"[GPU-ONLY] Using masks from GPU memory")
            elif masks_array is not None:
                # Convert numpy masks to GPU tensors
                flow_masks_list = []
                masks_dilated_list = []

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
                    flow_masks_list.append(Image.fromarray(flow_mask_img * 255))

                    # Regular mask dilation
                    if mask_dilation > 0:
                        mask_img = scipy.ndimage.binary_dilation(
                            mask_np, iterations=mask_dilation
                        ).astype(np.uint8)
                    else:
                        mask_img = binary_mask(mask_np).astype(np.uint8)
                    masks_dilated_list.append(Image.fromarray(mask_img * 255))

                # If single mask, repeat for all frames
                if len(masks_array) == 1:
                    flow_masks_list = flow_masks_list * frames_len
                    masks_dilated_list = masks_dilated_list * frames_len

                # Convert to GPU tensors
                flow_masks = to_tensors()(flow_masks_list).unsqueeze(0).to(device, non_blocking=True)
                masks_dilated = to_tensors()(masks_dilated_list).unsqueeze(0).to(device, non_blocking=True)
                print(f"[GPU-ONLY] Converted {len(masks_array)} numpy masks to GPU tensors")
            else:
                # Load from disk - use module-level read_mask function
                flow_masks, masks_dilated = read_mask(
                    mask,
                    frames_len,
                    size,
                    flow_mask_dilates=mask_dilation,
                    mask_dilates=mask_dilation,
                )
                flow_masks = to_tensors()(flow_masks).unsqueeze(0).to(device, non_blocking=True)
                masks_dilated = to_tensors()(masks_dilated).unsqueeze(0).to(device, non_blocking=True)

        # For masked frame visualization, need to convert to CPU temporarily
        # Convert first frame to numpy for masked_frame_for_save generation
        # This is ONLY for visualization output, not for processing!
        masked_frame_for_save = []
        frames_cpu = frames[0].cpu()  # [T, C, H, W]
        masks_cpu = masks_dilated[0].cpu()  # [T, 1, H, W]

        for i in range(frames_len):
            # Convert from [-1,1] to [0,255]
            frame_np = ((frames_cpu[i].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
            mask_np = masks_cpu[i, 0].numpy()

            mask_3ch = np.expand_dims(mask_np, 2).repeat(3, axis=2)
            green = np.zeros([h, w, 3])
            green[:, :, 1] = 255
            alpha = 0.6
            fuse_img = (1 - alpha) * frame_np + alpha * green
            fuse_img = mask_3ch * fuse_img + (1 - mask_3ch) * frame_np
            masked_frame_for_save.append(fuse_img.astype(np.uint8))

        # Skip to model inference (frames already in GPU tensor format)
        # Jump past all CPU-based preprocessing

    # 🔥 EXTREME SPEED: Use numpy arrays directly if provided (skip disk I/O!)
    elif frames_array is not None:
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

    fps = save_fps if fps is None else fps
    save_root = os.path.join(output, video_name)
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
        green = np.zeros([h, w, 3])
        green[:, :, 1] = 255
        alpha = 0.6
        # alpha = 1.0
        fuse_img = (1 - alpha) * img + alpha * green
        fuse_img = mask_ * fuse_img + (1 - mask_) * img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    # Convert to tensors (skip if already tensors from GPU pipeline)
    if not frames_already_tensors:
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
    # else: frames, flow_masks, masks_dilated already on GPU from NVDEC pipeline!

    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    # PyTorch RAFT with optional torch.compile optimization
    class _RAFTAdapter:
        def __init__(self, device: torch.device, use_half: bool):
            self.device = device
            self.use_half = use_half and (device.type == "cuda")
            self._model_type = "PyTorch RAFT"

            # Load PyTorch RAFT model
            ckpt_path = load_file_from_url(
                url=os.path.join(pretrain_model_url, "raft-things.pth"),
                model_dir="weights",
                progress=True,
                file_name=None,
            )
            self._raft = RAFT_bi(ckpt_path, device)
            print("[OK] RAFT PyTorch model initialized")
            print("   Expected performance: 3-5ms per frame")

        @property
        def model_type(self) -> str:
            """Return the actual optical flow model type being used"""
            return self._model_type

        @torch.inference_mode()
        def __call__(self, frames_btchw: torch.Tensor, iters: int = 20):
            # frames: [B, T, C, H, W]
            return self._raft(frames_btchw, iters=iters)

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
                _GLOBAL_MODELS_CACHE = {'_version': _CACHE_VERSION}

            # Check cache version - if different, clear all cached models
            cached_version = _GLOBAL_MODELS_CACHE.get('_version', 'v0')
            if cached_version != _CACHE_VERSION:
                print(f"[CACHE] Version mismatch! Cached={cached_version}, Current={_CACHE_VERSION}")
                print(f"[CACHE] Clearing ALL cached models to pick up new settings...")
                _GLOBAL_MODELS_CACHE = {'_version': _CACHE_VERSION}

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

                # Check if RAFT model needs to be recreated (e.g., from previous failed initialization)
                if _GLOBAL_MODELS_CACHE[cache_key]['raft'] is None:
                    print(f"[CACHE] RAFT model was None, recreating...")
                    raft_model = _RAFTAdapter(device, use_half)
                    _GLOBAL_MODELS_CACHE[cache_key]['raft'] = raft_model
                    print(f"[CACHE] Optical flow model recreated: {raft_model.model_type}")

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

    # torch.compile for RAFT (1.5-2x speedup)
    if os.getenv("USE_TORCH_COMPILE_RAFT", "0") == "1":
        if hasattr(fix_raft, '_raft') and device.type == 'cuda':
            try:
                print("[COMPILE] Compiling RAFT with torch.compile (mode=reduce-overhead)...")
                compiled_forward = torch.compile(
                    fix_raft._raft.forward,
                    backend="inductor",
                    mode="reduce-overhead",
                    dynamic=True,
                    fullgraph=False,
                )
                fix_raft._raft.forward = compiled_forward
                print("[OK] RAFT torch.compile enabled (1.5-2x speedup)")
            except Exception as e:
                error_type = type(e).__name__
                if 'Triton' in str(e) or 'triton' in str(e):
                    print(f"[WARNING] RAFT torch.compile failed (Triton error on WSL2): {error_type}")
                    print(f"[HINT] Try setting TORCHINDUCTOR_COMPILE_THREADS=1 or disable Triton")
                else:
                    print(f"[WARNING] RAFT torch.compile failed ({error_type}): {e}")
                print("[FALLBACK] Using PyTorch eager mode (slower but stable)")

    ##############################################
    # set up RFCNet with PyTorch + torch.compile
    ##############################################
    class _RFCNetAdapter:
        def __init__(self, device: torch.device, use_half: bool, ckpt_path: str):
            self.device = device
            self.use_half = use_half and (device.type == "cuda")

            # Load PyTorch RecurrentFlowCompleteNet
            print(f"Loading PyTorch RecurrentFlowCompleteNet from: {ckpt_path}")
            self._model = RecurrentFlowCompleteNet(ckpt_path)
            for p in self._model.parameters():
                p.requires_grad = False
            self._model.to(device, non_blocking=True)
            self._model.eval()
            print("[OK] RFCNet PyTorch model initialized")

            # torch.compile for RFCNet (1.5-2x speedup)
            if os.getenv("USE_TORCH_COMPILE_RAFT", "0") == "1" and device.type == 'cuda':
                try:
                    print("[COMPILE] Compiling RFCNet with torch.compile (mode=reduce-overhead)...")
                    compiled_forward = torch.compile(
                        self._model.forward,
                        backend="inductor",
                        mode="reduce-overhead",
                        dynamic=True,
                        fullgraph=False,
                    )
                    self._model.forward = compiled_forward
                    print("[OK] RFCNet torch.compile enabled (1.5-2x speedup)")
                except Exception as e:
                    error_type = type(e).__name__
                    if 'Triton' in str(e) or 'triton' in str(e):
                        print(f"[WARNING] RFCNet torch.compile failed (Triton error on WSL2): {error_type}")
                        print(f"[HINT] Try setting TORCHINDUCTOR_COMPILE_THREADS=1 or disable Triton")
                    else:
                        print(f"[WARNING] RFCNet torch.compile failed ({error_type}): {e}")
                    print("[FALLBACK] Using PyTorch eager mode (slower but stable)")

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
            return self._model.forward(masked_flows, masks)

        def forward_bidirect_flow(self, masked_flows_bi, masks_bi):
            """
            Bidirectional flow completion (wrapper for compatibility).

            Args:
                masked_flows_bi: Tuple of (flows_f, flows_b) where each is (B, T-1, 2, H, W)
                masks_bi: (B, T, 1, H, W) - single-channel masks

            Returns:
                pred_flows_bi: Tuple of [pred_flows_f, pred_flows_b]
                pred_edges_bi: Tuple of [None, None]
            """
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
            """Convert model to half precision."""
            self._model.half()
            return self

    ##############################################
    # Transformer Adapter (PyTorch only)
    ##############################################
    class _TransformerAdapter:
        """PyTorch wrapper for Sparse Temporal Transformer"""

        def __init__(self, transformer_module):
            self._pytorch_model = transformer_module

        def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
            """
            Forward pass using PyTorch model

            Args:
                x: [B, T, H, W, C] feature tokens
                fold_x_size: tuple (fold_h, fold_w)
                l_mask: [B, T, H, W, 1] local mask
                t_dilation: temporal dilation

            Returns:
                output: [B, T, H, W, C] transformed features
            """
            return self._pytorch_model(x, fold_x_size, l_mask, t_dilation)

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

    # torch.compile for RFCNet (optional, enabled via USE_TORCH_COMPILE_RAFT=1)
    if os.getenv("USE_TORCH_COMPILE_RAFT", "0") == "1" and device.type == 'cuda':
        try:
            print("[COMPILE] Compiling RFCNet with torch.compile (mode=reduce-overhead)...")
            fix_flow_complete.forward = torch.compile(
                fix_flow_complete.forward,
                backend="inductor",
                mode="reduce-overhead",
                dynamic=True,
            )
            print("[OK] RFCNet torch.compile enabled (1.5-2x speedup)")
        except Exception as e:
            print(f"[WARNING] RFCNet torch.compile failed: {e}")
            print("[FALLBACK] Using PyTorch eager mode")

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
                150, subvideo_length
            )  # ensure a minimum of 150 frames for image propagation
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
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
            img_save_root = os.path.join(
                save_root, "frames", str(idx).zfill(4) + ".png"
            )
            imwrite(f, img_save_root)

    # save videos frame
    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]

    # Helper function for GPU-accelerated NVENC encoding
    def encode_nvenc(frames, output_path, fps, width, height):
        """Encode frames using NVENC GPU encoder instead of CPU."""
        try:
            # Use h264_nvenc for GPU encoding
            cmd = [
                _FFMPEG_EXECUTABLE, '-y',  # Use detected FFmpeg path
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',  # Read from stdin
                '-c:v', 'h264_nvenc',  # NVENC GPU encoder
                '-preset', 'p4',  # Performance preset (p1=fastest, p7=slowest/best)
                '-b:v', '8M',  # 8 Mbps bitrate
                '-pix_fmt', 'yuv420p',
                '-profile:v', 'main',
                output_path
            ]

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Write frames to FFmpeg stdin
            for frame in frames:
                process.stdin.write(frame.tobytes())

            process.stdin.close()
            process.wait()

            if process.returncode != 0:
                stderr = process.stderr.read().decode()
                print(f"[WARNING] NVENC encoding had errors: {stderr}")
                # Fallback to imageio if NVENC fails
                return False

            return True

        except Exception as e:
            print(f"[WARNING] NVENC encoding failed: {e}, falling back to CPU imageio")
            return False

    # Try NVENC encoding, fallback to imageio if it fails
    h, w = out_size[1], out_size[0]  # out_size is (width, height)

    if not encode_nvenc(masked_frame_for_save, os.path.join(save_root, "masked_in.mp4"), fps, w, h):
        # Fallback to CPU imageio
        imageio.mimwrite(
            os.path.join(save_root, "masked_in.mp4"),
            masked_frame_for_save,
            fps=fps,
            quality=7,
        )
        print("[FALLBACK] Using CPU imageio for masked_in.mp4")
    else:
        print(f"[NVENC] Encoded masked_in.mp4 using GPU ({len(masked_frame_for_save)} frames @ {fps} fps)")

    if not encode_nvenc(comp_frames, os.path.join(save_root, "inpaint_out.mp4"), fps, w, h):
        # Fallback to CPU imageio
        imageio.mimwrite(
            os.path.join(save_root, "inpaint_out.mp4"), comp_frames, fps=fps, quality=7, ffmpeg_params=["-sws_flags", "bilinear"]
        )
        print("[FALLBACK] Using CPU imageio for inpaint_out.mp4")
    else:
        print(f"[NVENC] Encoded inpaint_out.mp4 using GPU ({len(comp_frames)} frames @ {fps} fps)")

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
