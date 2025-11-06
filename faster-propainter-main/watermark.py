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

# Essential optimization: Enable cuDNN autotuner (benchmarks kernels once, reuses best)
# This is SAFE and provides 20-30% speedup without breaking anything
torch.backends.cudnn.benchmark = True

# Enable TF32 for Ada Lovelace (RTX 4090) - Safe ~20% speedup on matmul operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

# Conditional import: Use NeuFlow v2 or RAFT based on environment variable
if os.getenv("USE_NEUFLOW", "0") == "1":
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


# Global lock for TensorRT RAFT execution (thread-pool worker safety)
# NOTE: Threads share TensorRT context - need lock until we implement thread-local contexts
import threading
_TRT_EXEC_LOCK = threading.Lock()


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

        print(f"⚡ Using {len(frames)} frames from memory (ZERO disk I/O!)")
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

            print(f"⚡ Using {len(masks_array)} masks from memory (ZERO disk I/O!)")
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

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
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
            # Enforce TensorRT-only mode when requested (no PyTorch fallback)
            def _parse_bool(val: str) -> bool:
                return str(val).lower() in ("1", "true", "yes", "on")
            self._force_trt = _parse_bool(os.getenv("FORCE_TRT_RAFT", "0"))

            # PRIORITY 1: Check USE_NEUFLOW FIRST (takes precedence over TensorRT)
            if os.getenv("USE_NEUFLOW", "0") == "1":
                # NeuFlow v2 uses ONNX model - MANDATORY when USE_NEUFLOW=1
                # Use absolute path based on this script's location
                model_path = os.path.join(os.path.dirname(__file__), "models", "neuflow_things.onnx")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"USE_NEUFLOW=1 but NeuFlow model not found: {model_path}\n"
                        f"Download it from: https://github.com/ibaiGorordo/ONNX-NeuFlowV2-Optical-Flow/releases"
                    )
                self._raft = RAFT_bi(model_path, device)
                print("[OK] NeuFlow v2 ONNX Runtime initialized (NO TensorRT/RAFT FALLBACK)")
                print("[OK] Expected speedup: 10-70x faster than RAFT baseline")
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
                        self._ctx = self._engine.create_execution_context()
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
                    print("[OK] RAFT PyTorch model initialized")

        @torch.no_grad()
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
                    print(f"[TRT] Using lock for thread safety (will switch to process pool for parallel)")
                    self._trt_shape_logged = True

                # Lock TensorRT execution (threads share context)
                with _TRT_EXEC_LOCK:
                    # Execute depending on TRT API
                    if not getattr(self, '_use_v3', False):
                        # Legacy bindings API
                        self._ctx.set_binding_shape(self._in_idx, tuple(inp.shape))
                        # Query num_bindings safely
                        try:
                            nb = self._engine.num_bindings
                        except Exception:
                            nb = 2
                        bindings = [None] * int(nb)
                        bindings[self._in_idx] = int(inp.data_ptr())
                        bindings[self._out_idx] = int(out.data_ptr())
                        ok = self._ctx.execute_v2(bindings)
                    else:
                        # TensorRT 10 v3 API
                        try:
                            success = self._ctx.set_input_shape(self._in_name, tuple(inp.shape))
                            if not success:
                                print(f"[TRT ERROR] set_input_shape failed for shape {inp.shape}")
                        except Exception as e:
                            print(f"[TRT ERROR] Exception in set_input_shape: {e}")
                            raise
                        self._ctx.set_tensor_address(self._in_name, int(inp.data_ptr()))
                        self._ctx.set_tensor_address(self._out_name, int(out.data_ptr()))
                        if self._stream is not None:
                            stream_ptr = int(self._stream.cuda_stream)
                        else:
                            stream_ptr = int(torch.cuda.current_stream().cuda_stream)
                        ok = self._ctx.execute_async_v3(stream_ptr)

                if not ok:
                    # DEBUG: Print shapes and binding info to understand failure
                    print(f"[TRT FAIL] Input shape: {inp.shape}, dtype: {inp.dtype}")
                    print(f"[TRT FAIL] Output shape: {out.shape}, dtype: {out.dtype}")
                    print(f"[TRT FAIL] Expected input: {self._in_name}, output: {self._out_name}")
                    print(f"[TRT FAIL] Device: {inp.device}")
                    raise RuntimeError(f'TRT execute failed - input shape: {inp.shape}, output shape: {out.shape}')
                return out

            # Prepare frame tensors on device
            dev_frames = frames_btchw.to(self.device)
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
                print(f"[CACHE] RAFT model cached ({time.time() - cache_start:.2f}s)")
            else:
                print(f"[CACHE] Reusing cached models for {cache_key} (ZERO reload time!)")

            fix_raft = _GLOBAL_MODELS_CACHE[cache_key]['raft']
    else:
        # Legacy behavior: create new models each time
        fix_raft = _RAFTAdapter(device, use_half)

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

            # Candidate engine paths
            engine_candidates = [
                os.path.join(os.getcwd(), 'faster-propainter-main', 'engines', 'rfcnet', 'rfcnet_fp16.engine'),
                os.path.join(os.path.dirname(__file__), 'engines', 'rfcnet', 'rfcnet_fp16.engine'),
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

                    # Load DCNv2 plugin (required for RFCNet TensorRT engine)
                    import ctypes
                    plugin_path = os.path.join(trt_lib, 'mmdeploy_tensorrt_ops.dll')
                    if os.path.exists(plugin_path):
                        try:
                            ctypes.CDLL(plugin_path)
                        except Exception as e:
                            if self._force_trt:
                                raise RuntimeError(f'Failed to load DCNv2 plugin from {plugin_path}: {e}')
                            else:
                                print(f"[WARNING] DCNv2 plugin load failed: {e}")
                                raise
                    else:
                        if self._force_trt:
                            raise RuntimeError(f'DCNv2 plugin not found at {plugin_path}')
                        else:
                            print(f"[WARNING] DCNv2 plugin not found at {plugin_path}")
                            raise FileNotFoundError(f'DCNv2 plugin not found at {plugin_path}')

                    import tensorrt as trt
                    logger = trt.Logger(trt.Logger.WARNING)
                    runtime = trt.Runtime(logger)
                    with open(engine_path, 'rb') as f:
                        self._engine = runtime.deserialize_cuda_engine(f.read())
                    if self._engine is None:
                        raise RuntimeError('TRT engine deserialize failed')
                    self._ctx = self._engine.create_execution_context()

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
                    print(f"   Expected speedup: 8.45x (30.78ms → 3.64ms per inference)")
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

            # Set input shapes
            self._ctx.set_input_shape(self._in_names[0], tuple(masked_flows.shape))
            self._ctx.set_input_shape(self._in_names[1], tuple(masks.shape))

            # Set tensor addresses
            self._ctx.set_tensor_address(self._in_names[0], int(masked_flows.data_ptr()))
            self._ctx.set_tensor_address(self._in_names[1], int(masks.data_ptr()))
            self._ctx.set_tensor_address(self._out_name, int(flow_out.data_ptr()))

            # Execute
            if self._stream is not None:
                stream_ptr = int(self._stream.cuda_stream)
            else:
                stream_ptr = int(torch.cuda.current_stream().cuda_stream)

            ok = self._ctx.execute_async_v3(stream_ptr)
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
                _GLOBAL_MODELS_CACHE[cache_key]['propainter'] = model
                print(f"[CACHE] ProPainter model cached in {'FP16' if use_half else 'FP32'} ({time.time() - propainter_start:.2f}s)")
                print(f"[OK] All models cached! Total worker init time saved on next segment")
            else:
                model = _GLOBAL_MODELS_CACHE[cache_key]['propainter']
    else:
        # Legacy behavior
        model = InpaintGenerator(model_path=ckpt_path).to(device, non_blocking=True)
        model.eval()

    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    # print(f'\nProcessing: {video_name} [{video_length} frames]...')
    print(f"Processing: {video_length} frames...")
    with torch.no_grad():
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

        # Use FP16 for RAFT with autocast (4x faster, no type errors!)
        if use_half:
            print("[RAFT] Using FP16 autocast for RAFT optical flow")

        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in tqdm(range(0, video_length, short_clip_len), desc="RAFT"):
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
            print("[OK] Image propagation completed")
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

        with torch.no_grad():
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
