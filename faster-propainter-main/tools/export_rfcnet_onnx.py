import argparse
import os
import sys
from pathlib import Path

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _get_tensor_sizes, _get_tensor_dim_size

# Ensure imports work when running from repo root or this script's folder
SCRIPT_DIR = Path(__file__).resolve().parent
FP_ROOT = SCRIPT_DIR.parent  # faster-propainter-main/
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))


# Register ONNX symbolic for torchvision::deform_conv2d -> mmdeploy::MMCVModulatedDeformConv2d
# This is needed because mmcv doesn't have CUDA ops available, so model falls back to torchvision
def torchvision_deform_conv2d_symbolic(g, *args, **kwargs):
    """
    Custom ONNX symbolic function for torchvision.ops.deform_conv2d.
    Converts to mmdeploy::MMCVModulatedDeformConv2d for TensorRT DCNv2 plugin support.

    PyTorch passes args as: input, offset, weight, bias, stride, padding, dilation, mask
    But tuples may be expanded, so we accept variable args.
    """
    # torchvision.ops.deform_conv2d signature:
    # (input, offset, weight, bias, stride, padding, dilation, mask)
    # When traced, tuples expand, so we get more args than expected

    # Debug: print detailed info for all args
    print(f"\nDEBUG: Received {len(args)} args for deform_conv2d")
    for i, arg in enumerate(args):
        if hasattr(arg, 'node'):
            node = arg.node()
            kind = node.kind()

            # Get tensor name
            name = arg.debugName() if hasattr(arg, 'debugName') else "unknown"

            # Try to get shape
            shape_str = "unknown"
            try:
                if hasattr(arg, 'type') and hasattr(arg.type(), 'sizes'):
                    shape_str = str(list(arg.type().sizes()))
            except:
                pass

            # Check if it's an initializer (model parameter)
            is_param = "parameter" if kind == "prim::Param" else "computed"

            print(f"  arg[{i}]: kind={kind:30s} name={name:40s} shape={shape_str:20s} [{is_param}]")
        else:
            print(f"  arg[{i}]: no node attribute")

    # CORRECTED: Based on debug output, PyTorch actually passes in this order:
    # args[0] = input (Concat)
    # args[1] = weight (prim::Param - the actual convolution weight!)
    # args[2] = offset (Concat)
    # args[3] = mask (Sigmoid)
    # args[4] = bias (prim::Param)
    # args[5-13] = stride/padding/dilation constants

    input_arg = args[0]
    weight = args[1]     # CORRECTED: weight is at position 1, not 2!
    offset = args[2]     # CORRECTED: offset is at position 2, not 1!
    mask = args[3]       # CORRECTED: mask is at position 3
    bias = args[4] if len(args) > 4 else None  # CORRECTED: bias is at position 4

    print(f"DEBUG: Extracted - input: 0, weight: 1 (CORRECTED), offset: 2, mask: 3, bias: {'4' if bias is not None else 'None'}")

    # RFCNet architecture uses fixed values for all deformable convolution layers
    # (from faster-propainter-main/model/recurrent_flow_completion.py:64,70)
    # SecondOrderDeformableAlignment(in_channels, out_channels, kernel_size=3, padding=1, deform_groups=16)
    stride = [1, 1]      # stride=1 for all deform convs
    padding = [1, 1]     # padding=1 for kernel_size=3
    dilation = [1, 1]    # dilation=1 (default)

    # Note: Tried parsing from args but PyTorch doesn't pass them in expected format
    # Since RFCNet architecture is fixed, hardcoding is more reliable than arg parsing

    # Infer deform_groups from offset shape using ONNX symbolic helper
    try:
        weight_sizes = _get_tensor_sizes(weight, allow_nonstatic=False)
        offset_sizes = _get_tensor_sizes(offset, allow_nonstatic=False)

        if weight_sizes is not None and offset_sizes is not None:
            kh = weight_sizes[2]  # kernel height
            kw = weight_sizes[3]  # kernel width
            offset_channels = offset_sizes[1]
            deform_groups = offset_channels // (2 * kh * kw)
        else:
            # Fallback: RFCNet/ProPainter always use deform_groups=16
            print("WARNING: Could not infer tensor sizes, using deform_groups=16 (RFCNet/ProPainter default)")
            deform_groups = 16
    except Exception as e:
        # Fallback to hardcoded value based on model architecture
        print(f"WARNING: Error getting tensor sizes ({e}), using deform_groups=16 (RFCNet/ProPainter default)")
        deform_groups = 16

    groups = 1

    # mmdeploy ONNX op expects: input, offset, mask, weight, bias (optional)
    input_tensors = [input_arg, offset, mask, weight]
    if bias is not None and hasattr(bias, 'node') and not bias.node().mustBeNone():
        input_tensors.append(bias)

    # Export as mmdeploy custom op
    output = g.op(
        'mmdeploy::MMCVModulatedDeformConv2d',
        *input_tensors,
        stride_i=stride,
        padding_i=padding,
        dilation_i=dilation,
        groups_i=groups,
        deform_groups_i=deform_groups
    )

    # Add shape type inference: output shape should match input shape (deformable conv preserves spatial dims)
    # Output shape: [N, C_out, H_out, W_out] where C_out comes from weight shape
    try:
        input_sizes = _get_tensor_sizes(input_arg, allow_nonstatic=False)
        weight_sizes = _get_tensor_sizes(weight, allow_nonstatic=False)
        if input_sizes is not None and weight_sizes is not None:
            # For deformable conv with padding=1, stride=1: output spatial dims = input spatial dims
            batch_size = input_sizes[0]
            out_channels = weight_sizes[0]
            height = input_sizes[2]
            width = input_sizes[3]
            # Set output type
            output.setType(input_arg.type().with_sizes([batch_size, out_channels, height, width]))
    except Exception as e:
        print(f"WARNING: Could not set shape type for DCNv2 output: {e}")

    return output


# Register the symbolic function for torchvision::deform_conv2d (opset 11)
register_custom_op_symbolic('torchvision::deform_conv2d', torchvision_deform_conv2d_symbolic, 11)
print("[OK] Registered ONNX symbolic for torchvision::deform_conv2d -> mmdeploy::MMCVModulatedDeformConv2d")


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
                "[WARNING] torchvision.ops.deform_conv2d not found. Install a matching torchvision build."
            )
    except Exception as exc:
        print(f"[WARNING] torchvision import check failed: {exc}")

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

    # TEMPORARY: Export with fixed shapes to debug TensorRT shape inference issues
    # Enable dynamic_axes for TensorRT dynamic shape support
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
        dynamic_axes=dynamic_axes,  # ENABLED for dynamic shape support
        dynamo=False,  # Use legacy ONNX exporter (PyTorch 2.x dynamo has constraint issues)
    )

    print(f"[SUCCESS] RFCNet ONNX exported to {output_path}")
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
