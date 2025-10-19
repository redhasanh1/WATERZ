"""
Export the RFCNet (Recurrent Flow Completion) module to ONNX with bounded
sequence length, suitable for TensorRT engine compilation.

The exporter loads the pretrained weights (default: web/weights/recurrent_flow_completion.pth),
freezes the module, pads variable-length sequences to a fixed MAX_T, and exports
three inputs:
    - flows_forward:  float32 [batch, T, 2, H, W]
    - flows_backward: float32 [batch, T, 2, H, W]
    - flow_masks:     float32 [batch, T + 1, 1, H, W]

Outputs:
    - completed_forward:  float32 [batch, T, 2, H, W]
    - completed_backward: float32 [batch, T, 2, H, W]

Usage example (PowerShell):
    python faster-propainter-main\\tools\\export_rfcnet_onnx.py ^
        --weights web/weights/recurrent_flow_completion.pth ^
        --output faster-propainter-main/engines/rfcnet/rfcnet.onnx ^
        --max-t 60 --height 256 --width 256
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

THIS_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = THIS_DIR.parent  # faster-propainter-main
REPO_ROOT = PACKAGE_ROOT.parent  # workspace root
for path in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
DEFAULT_WEIGHTS = REPO_ROOT / "web" / "weights" / "recurrent_flow_completion.pth"


def load_rfcnet_module(weights_path: Path, device: torch.device):
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from utils.download_util import load_file_from_url

    if not weights_path.exists():
        weights_path = Path(
            load_file_from_url(
                url="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
                model_dir="weights",
                file_name="recurrent_flow_completion.pth",
            )
        )
    model = RecurrentFlowCompleteNet(str(weights_path))
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    return model


class RFCNetONNXWrapper(torch.nn.Module):
    def __init__(self, model, max_t: int):
        super().__init__()
        self.model = model
        self.max_t = max_t

    def forward(self, flows_f, flows_b, flow_masks):
        """
        flows_f, flows_b: [B, T, 2, H, W]
        flow_masks:      [B, T+1, 1, H, W]
        """
        # Clamp sequence to max_t by padding/truncation.
        b, t, _, h, w = flows_f.shape
        if t > self.max_t:
            flows_f = flows_f[:, : self.max_t]
            flows_b = flows_b[:, : self.max_t]
            flow_masks = flow_masks[:, : self.max_t + 1]
        elif t < self.max_t:
            pad_len = self.max_t - t
            pad = torch.zeros((b, pad_len, 2, h, w), dtype=flows_f.dtype, device=flows_f.device)
            flows_f = torch.cat([flows_f, pad], dim=1)
            flows_b = torch.cat([flows_b, pad], dim=1)

            mask_pad = torch.zeros((b, pad_len, 1, h, w), dtype=flow_masks.dtype, device=flow_masks.device)
            flow_masks = torch.cat([flow_masks, mask_pad], dim=1)

        flows = (flows_f, flows_b)
        pred_flows, _ = self.model.forward_bidirect_flow(flows, flow_masks)
        completed = self.model.combine_flow(flows, pred_flows, flow_masks)
        return completed


def export_rfcnet(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_rfcnet_module(args.weights.resolve(), device)
    wrapper = RFCNetONNXWrapper(model, max_t=args.max_t).to(device)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_flows = torch.zeros(1, args.max_t, 2, args.height, args.width, device=device, dtype=torch.float32)
    dummy_masks = torch.zeros(1, args.max_t + 1, 1, args.height, args.width, device=device, dtype=torch.float32)

    dynamic_axes = {
        "flows_forward": {0: "batch", 1: "frames", 3: "height", 4: "width"},
        "flows_backward": {0: "batch", 1: "frames", 3: "height", 4: "width"},
        "flow_masks": {0: "batch", 1: "frames_plus_one", 3: "height", 4: "width"},
        "output_0": {0: "batch", 1: "frames", 3: "height", 4: "width"},
        "output_1": {0: "batch", 1: "frames", 3: "height", 4: "width"},
    }

    torch.onnx.export(
        wrapper,
        (dummy_flows, dummy_flows, dummy_masks),
        output_path.as_posix(),
        opset_version=args.opset,
        input_names=["flows_forward", "flows_backward", "flow_masks"],
        output_names=["completed_forward", "completed_backward"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    print(f"[RFCNet] ONNX exported to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RFCNet (Recurrent Flow Completion) to ONNX.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to RFCNet weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "engines" / "rfcnet" / "rfcnet.onnx",
        help="Destination ONNX file.",
    )
    parser.add_argument("--max-t", type=int, default=60, help="Maximum sequence length (frames).")
    parser.add_argument("--height", type=int, default=256, help="Spatial height used for dummy input.")
    parser.add_argument("--width", type=int, default=256, help="Spatial width used for dummy input.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (>= 16 required for grid_sample).")
    return parser.parse_args()


if __name__ == "__main__":
    export_rfcnet(parse_args())
