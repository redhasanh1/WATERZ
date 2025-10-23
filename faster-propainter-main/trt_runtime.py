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
    Try to accelerate RFCNet.forward using Torch-TensorRT Dynamo.

    This compiles only the forward(masked_flows, masks) path. Methods that
    call self.forward (e.g., forward_bidirect_flow) will benefit.

    Returns True if compilation succeeded and was installed into the model.
    """
    enabled = _parse_bool(os.getenv("RFCNET_TORCHTRT", "0"))
    if not enabled:
        return False

    if not (device.type == "cuda" and torch.cuda.is_available()):
        print("⚠️ RFCNet Torch-TensorRT skipped (CUDA not available)")
        return False

    try:
        import torch_tensorrt
    except Exception as exc:
        print(f"⚠️ RFCNet Torch-TensorRT not available: {exc}")
        return False

    # Prepare input specs with dynamic shapes
    dtype = torch.half if use_fp16 else torch.float
    min_shape_1 = (1, 1, 2, min_hw[1], min_hw[0])
    opt_shape_1 = (1, min(max_t, 8), 2, opt_hw[1], opt_hw[0])
    max_shape_1 = (1, max_t, 2, max_hw[1], max_hw[0])
    in1 = torch_tensorrt.Input(
        min_shape=min_shape_1, opt_shape=opt_shape_1, max_shape=max_shape_1, dtype=dtype
    )

    min_shape_2 = (1, 1, 1, min_hw[1], min_hw[0])
    opt_shape_2 = (1, min(max_t, 8), 1, opt_hw[1], opt_hw[0])
    max_shape_2 = (1, max_t, 1, max_hw[1], max_hw[0])
    in2 = torch_tensorrt.Input(
        min_shape=min_shape_2, opt_shape=opt_shape_2, max_shape=max_shape_2, dtype=dtype
    )

    try:
        compiled = torch_tensorrt.dynamo.compile(
            model,
            inputs=[in1, in2],
            enabled_precisions={torch.half} if use_fp16 else {torch.float},
        )
    except Exception as exc:
        print(f"⚠️ RFCNet Torch-TensorRT compile failed: {exc}")
        return False

    # Install compiled forward into the existing instance so that
    # forward_bidirect_flow → self.forward uses the accelerated path.
    def _compiled_forward(masked_flows: torch.Tensor, masks: torch.Tensor):
        return compiled(masked_flows, masks)

    model.forward = _compiled_forward  # type: ignore[assignment]
    print("✅ RFCNet forward accelerated via Torch-TensorRT (Dynamo)")
    return True

