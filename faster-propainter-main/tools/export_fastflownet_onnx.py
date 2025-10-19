"""
Export the FastFlowNet optical-flow backbone (used inside flow_comp_raft.py)
to an ONNX graph with dynamic height/width profiles that are compatible with
TensorRT. The exported model expects an input tensor shaped
[batch, sequence(=2), channels(=3), height, width] and produces the flow
tensor returned by ptlflow's FastFlowNet (`["flows"]` entry).

Example:
    python export_fastflownet_onnx.py ^
        --output ../../faster-propainter-main/engines/raft/fastflownet.onnx ^
        --min-shape 1x2x3x128x128 --opt-shape 1x2x3x256x256 --max-shape 1x2x3x640x640
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch


def _parse_shape(shape_str: str) -> Tuple[int, int, int, int, int]:
    """
    Parse shape strings like "1x2x3x256x256" into integer tuples.
    Order is (batch, seq, channels, height, width).
    """
    dims = shape_str.lower().split("x")
    if len(dims) != 5:
        raise argparse.ArgumentTypeError(
            f"Shape '{shape_str}' must have exactly 5 dimensions: BxTxCxHxW"
        )
    try:
        return tuple(int(d) for d in dims)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Shape '{shape_str}' contains a non-integer value"
        ) from exc


def build_model(device: torch.device):
    """
    Load the pretrained FastFlowNet model via ptlflow and wrap it so that it
    accepts a torch tensor directly, matching how flow_comp_raft.py uses it.
    """
    try:
        import ptlflow
    except ImportError as exc:
        raise RuntimeError(
            "ptlflow is required for FastFlowNet export. Install with "
            "`pip install ptlflow` before running this script."
        ) from exc

    inner_model = ptlflow.get_model("fastflownet", ckpt_path="things")
    inner_model.to(device)
    inner_model.eval()

    class FastFlowNetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            # Input images: [B, 2, 3, H, W]
            outputs = self.model({"images": images})
            flows = outputs["flows"]
            # FastFlowNet returns [B, 1, 2, H, W]; squeeze the sequence dim to
            # align with flow_comp_raft expectations (forward/backward pair).
            if flows.dim() == 5 and flows.size(1) == 1:
                flows = flows.squeeze(1)
            return flows

    return FastFlowNetWrapper(inner_model)


def export_fastflownet(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_shape = args.opt_shape or args.min_shape or (1, 2, 3, 256, 256)
    dummy_input = torch.zeros(dummy_shape, dtype=torch.float32, device=device)

    dynamic_axes = {
        "images": {0: "batch", 3: "height", 4: "width"},
        "flows": {0: "batch", 2: "height", 3: "width"},
    }

    torch.onnx.export(
        model,
        dummy_input,
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["flows"],
        dynamic_axes=dynamic_axes,
    )

    print(f"[OK] FastFlowNet ONNX exported to {output_path}")
    if args.min_shape or args.opt_shape or args.max_shape:
        print(
            "   Exported with dynamic axes; remember to pass matching "
            "min/opt/max shapes when building the TensorRT engine."
        )


def main():
    parser = argparse.ArgumentParser(description="Export FastFlowNet to ONNX.")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            "..",
            "engines",
            "raft",
            "fastflownet.onnx",
        ),
        help="Path to save the ONNX model (directories created automatically).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset to use (>= 16 required for grid_sample; 17 recommended).",
    )
    parser.add_argument(
        "--min-shape",
        type=_parse_shape,
        default=None,
        help="Optional BxTxCxHxW string describing the minimum profile. "
        "Used only for logging; TensorRT profiles are set during engine build.",
    )
    parser.add_argument(
        "--opt-shape",
        type=_parse_shape,
        default=None,
        help="Optional BxTxCxHxW shape describing the optimal profile.",
    )
    parser.add_argument(
        "--max-shape",
        type=_parse_shape,
        default=None,
        help="Optional BxTxCxHxW shape describing the maximum profile.",
    )

    args = parser.parse_args()
    export_fastflownet(args)


if __name__ == "__main__":
    main()
