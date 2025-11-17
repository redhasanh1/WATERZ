import warnings
import os
import sys

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    # Enable Triton with per-worker cache isolation (WSL2 Linux only)
    # Use separate cache dir per process to avoid thread corruption
    if os.getenv('ENABLE_TRITON_KERNELS', '1') == '1' and sys.platform == 'linux':
        os.environ['TRITON_CACHE_DIR'] = f'/tmp/triton_cache_worker_{os.getpid()}'
        _TRITON_AVAILABLE = True
        print(f"[Triton] ✅ Enabled with isolated cache: /tmp/triton_cache_worker_{os.getpid()}")
    else:
        # DISABLED: Triton autotuner is NOT thread-safe on Windows
        _TRITON_AVAILABLE = False
        print("[Triton] ⚠️ Disabled (Windows or ENABLE_TRITON_KERNELS=0)")
except ImportError:  # pragma: no cover - only executed when Triton is missing.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def triton_is_available() -> bool:
    """Return ``True`` if the Triton package can be imported."""
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 16, "BLOCK_W": 16}, num_warps=4),
            triton.Config({"BLOCK_H": 32, "BLOCK_W": 32}, num_warps=8),
            triton.Config({"BLOCK_H": 64, "BLOCK_W": 64}, num_warps=16),
        ],
        key=["H", "W"],
    )
    @triton.jit
    def fused_flow_warp_mask_kernel(  # type: ignore[misc]
        frame_ptr,
        flow_ptr,
        mask_ptr,
        warped_masked_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        C: tl.constexpr,
        frame_stride_c,
        frame_stride_h,
        frame_stride_w,
        flow_stride_c,
        flow_stride_h,
        flow_stride_w,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Fuses flow warping, bilinear interpolation, and mask application."""
        pid_c = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_w = tl.program_id(2)

        h_start = pid_h * BLOCK_H
        w_start = pid_w * BLOCK_W

        h_offsets = h_start + tl.arange(0, BLOCK_H)
        w_offsets = w_start + tl.arange(0, BLOCK_W)

        h_mask = h_offsets < H
        w_mask = w_offsets < W

        flow_h_idx = (
            flow_stride_c * 0
            + flow_stride_h * h_offsets[:, None]
            + flow_stride_w * w_offsets[None, :]
        )
        flow_w_idx = (
            flow_stride_c * 1
            + flow_stride_h * h_offsets[:, None]
            + flow_stride_w * w_offsets[None, :]
        )

        flow_h = tl.load(
            flow_ptr + flow_h_idx,
            mask=h_mask[:, None] & w_mask[None, :],
            other=0.0,
        )
        flow_w = tl.load(
            flow_ptr + flow_w_idx,
            mask=h_mask[:, None] & w_mask[None, :],
            other=0.0,
        )

        src_h = h_offsets[:, None].to(tl.float32) + flow_h
        src_w = w_offsets[None, :].to(tl.float32) + flow_w

        h0 = tl.floor(src_h).to(tl.int32)
        w0 = tl.floor(src_w).to(tl.int32)
        h1 = h0 + 1
        w1 = w0 + 1

        wh = src_h - h0.to(tl.float32)
        ww = src_w - w0.to(tl.float32)

        valid_00 = (h0 >= 0) & (h0 < H) & (w0 >= 0) & (w0 < W)
        valid_01 = (h0 >= 0) & (h0 < H) & (w1 >= 0) & (w1 < W)
        valid_10 = (h1 >= 0) & (h1 < H) & (w0 >= 0) & (w0 < W)
        valid_11 = (h1 >= 0) & (h1 < H) & (w1 >= 0) & (w1 < W)

        frame_c_offset = frame_stride_c * pid_c

        idx_00 = frame_c_offset + frame_stride_h * h0 + frame_stride_w * w0
        idx_01 = frame_c_offset + frame_stride_h * h0 + frame_stride_w * w1
        idx_10 = frame_c_offset + frame_stride_h * h1 + frame_stride_w * w0
        idx_11 = frame_c_offset + frame_stride_h * h1 + frame_stride_w * w1

        val_00 = tl.load(frame_ptr + idx_00, mask=valid_00, other=0.0)
        val_01 = tl.load(frame_ptr + idx_01, mask=valid_01, other=0.0)
        val_10 = tl.load(frame_ptr + idx_10, mask=valid_10, other=0.0)
        val_11 = tl.load(frame_ptr + idx_11, mask=valid_11, other=0.0)

        top = val_00 * (1 - ww) + val_01 * ww
        bottom = val_10 * (1 - ww) + val_11 * ww
        warped = top * (1 - wh) + bottom * wh

        mask_idx = h_offsets[:, None] * W + w_offsets[None, :]
        mask_val = tl.load(
            mask_ptr + mask_idx,
            mask=h_mask[:, None] & w_mask[None, :],
            other=0.0,
        )

        result = warped * mask_val

        out_idx = (
            pid_c * H * W + h_offsets[:, None] * W + w_offsets[None, :]
        )

        tl.store(
            warped_masked_ptr + out_idx,
            result,
            mask=h_mask[:, None] & w_mask[None, :],
        )


def _grid_sample_fallback(
    frame: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """PyTorch fallback that mimics the fused kernel semantics."""
    if frame.ndim != 3:
        raise ValueError("Fallback expects frame shaped as [C, H, W].")
    if flow.shape != (2, frame.shape[1], frame.shape[2]):
        raise ValueError("Flow must have shape [2, H, W].")
    if mask.shape[-2:] != frame.shape[-2:]:
        raise ValueError("Mask spatial size must match frame.")

    device = frame.device
    dtype = frame.dtype
    C, H, W = frame.shape

    flow_hw = flow.permute(1, 2, 0).to(torch.float32)  # [H, W, 2]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device, dtype=flow_hw.dtype),
        torch.arange(W, device=device, dtype=flow_hw.dtype),
        indexing="ij",
    )
    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = base_grid + flow_hw

    if W > 1:
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    else:
        grid[..., 0] = 0.0
    if H > 1:
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
    else:
        grid[..., 1] = 0.0

    grid = grid.unsqueeze(0).to(dtype=torch.float32)
    frame_batched = frame.unsqueeze(0)

    warped = F.grid_sample(
        frame_batched,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    mask_tensor = mask.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    result = warped * mask_tensor
    return result.squeeze(0)


def fused_flow_warp_mask_single(
    frame: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Apply the fused flow warp + mask operation to a single sample.

    Args:
        frame: Tensor with shape ``[C, H, W]`` located on GPU or CPU.
        flow: Tensor with shape ``[2, H, W]`` matching ``frame``.
        mask: Tensor with shape ``[H, W]`` where 1 selects warped pixels.
        use_triton: When ``True`` and the Triton kernel is available, run it.
            Otherwise the function falls back to a PyTorch implementation.

    Returns:
        Tensor with shape ``[C, H, W]`` containing the warped-and-masked data.
    """
    if frame.ndim != 3:
        raise ValueError("frame must be 3D (C, H, W).")
    if flow.shape != (2, frame.shape[1], frame.shape[2]):
        raise ValueError("flow must have shape (2, H, W).")
    if mask.shape[-2:] != frame.shape[-2:]:
        raise ValueError("mask must align spatially with frame.")

    frame_c = frame.contiguous()
    flow_c = flow.contiguous()
    mask_c = mask.contiguous()

    can_use_triton = (
        use_triton
        and triton_is_available()
        and frame_c.is_cuda
        and flow_c.is_cuda
        and mask_c.is_cuda
    )

    if not can_use_triton:
        if use_triton and not triton_is_available():
            warnings.warn(
                "Triton is not available; falling back to PyTorch implementation.",
                RuntimeWarning,
            )
        return _grid_sample_fallback(frame_c, flow_c, mask_c)

    if frame_c.dtype not in (torch.float16, torch.float32):
        raise TypeError("frame must be float16 or float32 for Triton kernel.")

    output = torch.empty_like(frame_c)

    grid = lambda meta: (
        frame_c.shape[0],
        triton.cdiv(frame_c.shape[1], meta["BLOCK_H"]),
        triton.cdiv(frame_c.shape[2], meta["BLOCK_W"]),
    )

    fused_flow_warp_mask_kernel[grid](  # type: ignore[index]
        frame_c,
        flow_c,
        mask_c,
        output,
        frame_c.shape[1],
        frame_c.shape[2],
        frame_c.shape[0],
        frame_c.stride(0),
        frame_c.stride(1),
        frame_c.stride(2),
        flow_c.stride(0),
        flow_c.stride(1),
        flow_c.stride(2),
    )

    return output


def fused_flow_warp_mask(
    frame: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
    *,
    use_triton: bool = True,
) -> torch.Tensor:
    """Batch-aware wrapper around :func:`fused_flow_warp_mask_single`.

    Supports input tensors with shapes:

    - frame ``[C, H, W]`` / flow ``[2, H, W]`` / mask ``[H, W]``
    - frame ``[B, C, H, W]`` / flow ``[B, 2, H, W]`` / mask ``[B, H, W]`` or
      ``[B, 1, H, W]``
    """
    if frame.ndim == 3:
        return fused_flow_warp_mask_single(frame, flow, mask, use_triton=use_triton)

    if frame.ndim != 4:
        raise ValueError("frame must be 3D or 4D tensor.")
    if flow.ndim != 4 or flow.shape[0] != frame.shape[0]:
        raise ValueError("flow must be batched with shape [B, 2, H, W].")

    if mask.ndim == 4:
        mask_batched = mask.squeeze(1)
    elif mask.ndim == 3:
        mask_batched = mask
    else:
        raise ValueError("mask must have shape [B, H, W] or [B, 1, H, W].")

    outputs = []
    for idx in range(frame.shape[0]):
        outputs.append(
            fused_flow_warp_mask_single(
                frame[idx],
                flow[idx],
                mask_batched[idx],
                use_triton=use_triton,
            )
        )
    return torch.stack(outputs, dim=0)
