import torch

from model.modules.flow_loss_utils import flow_warp
from triton_ops import fused_flow_warp_mask


def reference_warp_mask(
    frame: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation mirroring the fused kernel."""
    if frame.ndim != 3:
        raise ValueError("Expected frame shaped [C, H, W].")
    frame_batched = frame.unsqueeze(0)
    flow_batched = flow.permute(1, 2, 0).unsqueeze(0)
    warped = flow_warp(frame_batched, flow_batched)
    masked = warped * mask.to(dtype=warped.dtype).unsqueeze(0).unsqueeze(0)
    return masked.squeeze(0)


def test_single_sample_cpu():
    torch.manual_seed(0)
    C, H, W = 3, 8, 8
    frame = torch.randn(C, H, W, dtype=torch.float32)
    flow = torch.randn(2, H, W, dtype=torch.float32) * 0.1
    mask = torch.rand(H, W, dtype=torch.float32).round()  # binary mask

    expected = reference_warp_mask(frame, flow, mask)
    result = fused_flow_warp_mask(frame, flow, mask, use_triton=False)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_batched_cpu():
    torch.manual_seed(1)
    B, C, H, W = 2, 4, 6, 6
    frame = torch.randn(B, C, H, W, dtype=torch.float32)
    flow = torch.randn(B, 2, H, W, dtype=torch.float32) * 0.1
    mask = torch.rand(B, 1, H, W, dtype=torch.float32).round()

    expected = []
    for idx in range(B):
        expected.append(
            reference_warp_mask(frame[idx], flow[idx], mask[idx, 0])
        )
    expected = torch.stack(expected, dim=0)

    result = fused_flow_warp_mask(frame, flow, mask, use_triton=False)
    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_single_sample_cpu()
    test_batched_cpu()
    print("✅ Triton fused kernel fallback tests passed.")
