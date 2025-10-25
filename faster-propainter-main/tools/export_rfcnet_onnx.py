import argparse
import os
import sys
from pathlib import Path

import torch


# Ensure imports work when running from repo root or this script's folder
SCRIPT_DIR = Path(__file__).resolve().parent
FP_ROOT = SCRIPT_DIR.parent  # faster-propainter-main/
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))


def build_model(weights_path: str):
    try:
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet  # type: ignore
    except ModuleNotFoundError:
        # Fallback: import module directly from file path
        import importlib.util as _ilu
        rfc_path = FP_ROOT / "model" / "recurrent_flow_completion.py"
        if not rfc_path.exists():
            raise
        spec = _ilu.spec_from_file_location("model.recurrent_flow_completion", rfc_path.as_posix())
        if spec is None or spec.loader is None:
            raise
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        RecurrentFlowCompleteNet = getattr(mod, "RecurrentFlowCompleteNet")  # type: ignore

    # Resolve weights robustly if a relative path is provided
    wp = Path(weights_path)
    if not wp.is_absolute() and not wp.exists():
        # Try repo root weights/
        candidate = FP_ROOT.parent / "weights" / Path(weights_path).name
        if candidate.exists():
            wp = candidate

    model = RecurrentFlowCompleteNet(model_path=str(wp))
    model.eval()
    return model


def _parse_shape_5d(shape_str: str):
    # Expect BxTxCxHxW
    parts = shape_str.lower().replace(" ", "").split("x")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"Shape must be BxTxCxHxW (got: {shape_str})"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Shape contains non-integer values: {shape_str}"
        )


def export_rfcnet_onnx(args: argparse.Namespace) -> None:
    # Preflight for torchvision.ops.deform_conv2d
    try:
        import torchvision  # noqa: F401
        if not hasattr(torchvision.ops, "deform_conv2d"):
            print(
                "⚠️ torchvision.ops.deform_conv2d not found. Install a matching torchvision build."
            )
    except Exception as exc:
        print(f"⚠️ torchvision import check failed: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inner = build_model(args.weights)
    inner.to(device)

    class RFCNetExportWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module, export_edges: bool = False):
            super().__init__()
            self.model = model
            self.export_edges = export_edges

        def forward(self, masked_flows: torch.Tensor, masks: torch.Tensor):
            flow, edge = self.model(masked_flows, masks)
            if self.export_edges:
                if edge is None:
                    b, t, _, h, w = flow.shape
                    edge = torch.zeros((b, t, 1, h, w), dtype=flow.dtype, device=flow.device)
                return flow, edge
            return flow

    model = RFCNetExportWrapper(inner, export_edges=args.export_edges)
    model.eval()

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy inputs match forward signature: (masked_flows, masks)
    # masked_flows: [B, T, 2, H, W]
    # masks: [B, T, 1, H, W]  ← NOTE: 1 channel mask (matches runtime usage)
    b, t, c, h, w = args.optshape or args.minshape or (1, 8, 2, 256, 256)
    dummy_masked_flows = torch.zeros((b, t, 2, h, w), dtype=torch.float32, device=device)
    dummy_masks = torch.zeros((b, t, 1, h, w), dtype=torch.float32, device=device)

    dynamic_axes = {
        "masked_flows": {0: "batch", 1: "time", 3: "height", 4: "width"},
        "masks": {0: "batch", 1: "time", 3: "height", 4: "width"},
        "flow": {0: "batch", 1: "time", 3: "height", 4: "width"},
    }
    output_names = ["flow"]
    if args.export_edges:
        dynamic_axes["edge"] = {0: "batch", 1: "time", 3: "height", 4: "width"}
        output_names = ["flow", "edge"]

    torch.onnx.export(
        model,
        (dummy_masked_flows, dummy_masks),
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["masked_flows", "masks"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy ONNX exporter (PyTorch 2.x dynamo has constraint issues)
    )

    print(f"✅ RFCNet ONNX exported to {output_path}")
    print(
        "   Note: This model uses torchvision.ops.deform_conv2d (DCNv2). "
        "For TensorRT, you may need a DCNv2 plugin or keep that op in a Torch fallback."
    )

    if args.minshape or args.optshape or args.maxshape:
        print(
            "   Exported with dynamic axes; remember to pass matching min/opt/max "
            "shapes when building the TensorRT engine with trtexec."
        )


def main():
    parser = argparse.ArgumentParser(description="Export RFCNet (flow completion) to ONNX.")
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join("weights", "recurrent_flow_completion.pth"),
        help="Path to RFCNet weights (downloaded by runtime if missing).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join(
            "..",
            "engines",
            "rfcnet",
            "rfcnet.onnx",
        ),
        help="Path to save the ONNX model.",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--minshape",
        type=_parse_shape_5d,
        default=None,
        help="Optional BxTxCxHxW minimum profile for logging.",
    )
    parser.add_argument(
        "--optshape",
        type=_parse_shape_5d,
        default=None,
        help="Optional BxTxCxHxW optimal profile for logging.",
    )
    parser.add_argument(
        "--maxshape",
        type=_parse_shape_5d,
        default=None,
        help="Optional BxTxCxHxW maximum profile for logging.",
    )
    parser.add_argument(
        "--export-edges",
        action="store_true",
        help="Export the edge output as well (used during training).",
    )

    args = parser.parse_args()
    export_rfcnet_onnx(args)


if __name__ == "__main__":
    main()
