import os
from typing import Tuple

import torch


def _parse_bool(val: str) -> bool:
    return str(val).lower() in ("1", "true", "yes", "on")


def maybe_compile_rfcnet(
    model: torch.nn.Module,
    device: torch.device,
    *,
    use_fp16: bool = True,
    min_hw: Tuple[int, int] = (128, 128),
    opt_hw: Tuple[int, int] = (480, 480),
    max_hw: Tuple[int, int] = (640, 640),
    max_t: int = 12,
) -> bool:
    """
    Try to accelerate RFCNet.forward using torch.compile() with inductor backend.

    This compiles only the forward(masked_flows, masks) path. Methods that
    call self.forward (e.g., forward_bidirect_flow) will benefit.

    Returns True if compilation succeeded and was installed into the model.
    """
    enabled = _parse_bool(os.getenv("RFCNET_TORCHTRT", "0"))
    if not enabled:
        return False

    if not (device.type == "cuda" and torch.cuda.is_available()):
        print("[WARNING] RFCNet torch.compile skipped (CUDA not available)")
        return False

    try:
        # Use torch.compile() with inductor backend (PyTorch's built-in optimizer)
        compiled_forward = torch.compile(
            model.forward,
            backend="inductor",
            mode="max-autotune",
            dynamic=True,
        )
        # Install compiled forward into the existing instance
        model.forward = compiled_forward  # type: ignore[assignment]
    except Exception as exc:
        print(f"[WARNING] RFCNet torch.compile(inductor) failed: {exc}")
        return False
    print("[OK] RFCNet forward accelerated via torch.compile(backend='inductor')")
    return True

