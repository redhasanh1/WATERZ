"""
Export RFC Net with DCNv4 to ONNX for TensorRT Plugin

This script exports RFC Net with DCNv4 deformable convolution as a custom ONNX op
that will be handled by our TensorRT DCNv4 plugin (3x faster than DCNv2).

Usage:
    python export_rfcnet_dcnv4_onnx.py --weights weights/recurrent_flow_completion.pth \\
                                       --out engines/rfcnet/rfcnet_dcnv4.onnx
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import _get_tensor_sizes

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
FP_ROOT = SCRIPT_DIR / "faster-propainter-main"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))

# Add DCNv4 to path
DCNv4_ROOT = SCRIPT_DIR / "DCNv4" / "DCNv4_op"
if str(DCNv4_ROOT) not in sys.path:
    sys.path.insert(0, str(DCNv4_ROOT))

# CRITICAL: Enable DCNv4 before importing model
os.environ['ENABLE_DCNV4_RFCNET'] = '1'
# Disable FP8 for ONNX export (causes type mismatches during tracing)
# FP8 will be applied during TensorRT engine building instead
os.environ['ENABLE_FP8_RFCNET'] = '0'

print("=" * 80)
print("RFC NET WITH DCNv4 ONNX EXPORT")
print("=" * 80)
print(f"  ENABLE_DCNV4_RFCNET = {os.environ['ENABLE_DCNV4_RFCNET']}")
print(f"  ENABLE_FP8_RFCNET = {os.environ['ENABLE_FP8_RFCNET']} (disabled for ONNX export)")
print()


# ============================================================================
# DCNv4 ONNX Symbolic Function
# ============================================================================

def dcnv4_forward_symbolic(g, features, offsets,
                           kernel_h, kernel_w,
                           stride_h, stride_w,
                           pad_h, pad_w,
                           dilation_h, dilation_w,
                           group, group_channels,
                           offset_scale, block_thread, remove_center, **kwargs):
    """
    Register DCNv4Function.apply as custom ONNX op for TensorRT plugin.

    DCNv4Function.apply signature:
    - features: [N, H, W, C] feature maps
    - offsets: [N, H, W, G*K*3] offset/mask tensor (K=kernel_size^2)
    - Plus various parameters (kernel_size, stride, pad, etc.)

    TensorRT plugin expects:
    - Input 0: Features [N, C, H, W] FP16
    - Input 1: Offsets [N, G*K*3, H, W] FP16
    - Output: [N, C, H, W] FP16

    This function converts NHWC → NCHW format for both inputs.

    Args:
        g: ONNX graph
        features: [N, H, W, C] tensor
        offsets: [N, H, W, G*K*3] tensor
        kernel_h, kernel_w: Kernel size
        stride_h, stride_w: Stride
        pad_h, pad_w: Padding
        dilation_h, dilation_w: Dilation
        group: Number of groups
        group_channels: Channels per group (D)
        offset_scale: Offset scaling factor
        block_thread: CUDA block size (ignored in ONNX)
        remove_center: Remove center pixel

    Returns:
        ONNX node for custom::DCNv4Plugin
    """
    print(f"\n[DCNv4 Symbolic] Registering DCNv4Function.apply as custom::DCNv4Plugin")
    print(f"  Features: {features.debugName() if hasattr(features, 'debugName') else 'unknown'}")
    print(f"  Offsets: {offsets.debugName() if hasattr(offsets, 'debugName') else 'unknown'}")

    # Extract shape information from features [N, H, W, C]
    try:
        feature_sizes = _get_tensor_sizes(features, allow_nonstatic=False)
        if feature_sizes is not None:
            N, H, W, C = feature_sizes
            print(f"  Feature shape: [N={N}, H={H}, W={W}, C={C}]")
        else:
            print(f"  Feature shape: Dynamic (will be inferred at runtime)")
            N, H, W, C = None, None, None, None
    except Exception as e:
        print(f"  WARNING: Could not get feature sizes: {e}")
        N, H, W, C = None, None, None, None

    # Extract offset dimensions
    try:
        offset_sizes = _get_tensor_sizes(offsets, allow_nonstatic=False)
        if offset_sizes is not None:
            N_off, H_off, W_off, offset_dim = offset_sizes
            print(f"  Offset shape: [N={N_off}, H={H_off}, W={W_off}, offset_dim={offset_dim}]")
        else:
            offset_dim = None
    except Exception as e:
        print(f"  WARNING: Could not get offset sizes: {e}")
        offset_dim = None

    # Convert NHWC -> NCHW for features
    # [N, H, W, C] -> [N, C, H, W]
    features_nchw = g.op('Transpose', features, perm_i=[0, 3, 1, 2])
    print(f"  Transposed features: NHWC -> NCHW")

    # Convert NHWC -> NCHW for offsets
    # [N, H, W, G*K*3] -> [N, G*K*3, H, W]
    offsets_nchw = g.op('Transpose', offsets, perm_i=[0, 3, 1, 2])
    print(f"  Transposed offsets: NHWC -> NCHW")

    # Calculate total channels (group * group_channels)
    total_channels = group * group_channels if isinstance(group, int) and isinstance(group_channels, int) else (C if C is not None else 128)

    # Create custom ONNX op with 2 inputs
    output_nchw = g.op(
        'custom::DCNv4Plugin',
        features_nchw,
        offsets_nchw,
        channels_i=int(total_channels),
        kernel_size_i=int(kernel_h),  # Use kernel_h (should equal kernel_w)
        stride_i=int(stride_h),       # Use stride_h (should equal stride_w)
        pad_i=int(pad_h),            # Use pad_h (should equal pad_w)
        group_i=int(group),
        offset_scale_f=float(offset_scale),
        remove_center_i=int(remove_center),
    )
    print(f"  [OK] Created custom::DCNv4Plugin node with 2 inputs")

    # Convert output back from NCHW -> NHWC
    # [N, C, H, W] -> [N, H, W, C]
    output_nhwc = g.op('Transpose', output_nchw, perm_i=[0, 2, 3, 1])
    print(f"  Transposed output: NCHW -> NHWC")

    # Set output shape: [N, H, W, C] (same as input for DCNv4)
    if N is not None and H is not None and W is not None and C is not None:
        output_nhwc.setType(features.type().with_sizes([N, H, W, C]))
        print(f"  Output shape: [N={N}, H={H}, W={W}, C={C}]")

    print(f"  [OK] DCNv4Plugin registered:")
    print(f"    - channels={total_channels}, kernel_size={kernel_h}, group={group}")
    print(f"    - stride={stride_h}, pad={pad_h}, offset_scale={offset_scale}")
    return output_nhwc


# Register DCNv4 symbolic function
# The key is to register for the actual DCNv4 function call
# We need to find what namespace DCNv4 uses when traced
# Based on DCNv4 implementation, it's likely "DCNv4::forward" or similar

# Try multiple registration strategies
# PyTorch traces autograd Functions using their module path
registration_strategies = [
    'DCNv4::DCNv4Function',   # Function class name
    'DCNv4::forward',         # Function forward method
    'aten::DCNv4',            # aten namespace (if uses custom ops)
    'prim::PythonOp',         # Fallback for Python ops
]

for strategy in registration_strategies:
    try:
        register_custom_op_symbolic(strategy, dcnv4_forward_symbolic, 12)
        print(f"[OK] Registered ONNX symbolic: {strategy} -> custom::DCNv4Plugin")
    except Exception as e:
        print(f"[WARNING] Could not register {strategy}: {e}")


# ============================================================================
# Model Builder
# ============================================================================

def build_model(weights_path: str):
    """Load RFC Net model with DCNv4 enabled."""
    try:
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet, ENABLE_DCNV4_RFCNET
    except ModuleNotFoundError:
        import importlib.util as _ilu
        rfc_path = FP_ROOT / "model" / "recurrent_flow_completion.py"
        if not rfc_path.exists():
            raise RuntimeError(f"RFC Net model not found at {rfc_path}")

        spec = _ilu.spec_from_file_location("model.recurrent_flow_completion", rfc_path.as_posix())
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load RFC Net module")

        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        RecurrentFlowCompleteNet = getattr(mod, "RecurrentFlowCompleteNet")
        ENABLE_DCNV4_RFCNET = getattr(mod, "ENABLE_DCNV4_RFCNET")

    # Verify DCNv4 is enabled
    if not ENABLE_DCNV4_RFCNET:
        raise RuntimeError(
            "DCNv4 is not enabled! Set ENABLE_DCNV4_RFCNET=1 before importing model.\n"
            "This script should have done this automatically - check DCNv4 installation."
        )

    print(f"[OK] DCNv4 is enabled: ENABLE_DCNV4_RFCNET={ENABLE_DCNV4_RFCNET}")

    # Resolve weights path
    wp = Path(weights_path)
    if not wp.is_absolute() and not wp.exists():
        candidate = Path("weights") / wp.name
        if candidate.exists():
            wp = candidate

    if not wp.exists():
        print(f"[WARNING] Weights not found at {wp}, using random initialization")
        model = RecurrentFlowCompleteNet()
    else:
        model = RecurrentFlowCompleteNet(model_path=str(wp))

    model.eval()

    # Verify DCNv4 layers are present
    from model.recurrent_flow_completion import SecondOrderDeformableAlignmentDCNv4
    dcnv4_count = sum(1 for m in model.modules() if isinstance(m, SecondOrderDeformableAlignmentDCNv4))
    print(f"[OK] Found {dcnv4_count} DCNv4 deformable alignment layers")

    if dcnv4_count == 0:
        raise RuntimeError("No DCNv4 layers found in model! Check ENABLE_DCNV4_RFCNET flag.")

    return model


def _parse_shape_5d(shape_str: str):
    """Parse BxTxCxHxW shape string."""
    parts = shape_str.lower().replace(" ", "").split("x")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(f"Shape must be BxTxCxHxW (got: {shape_str})")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Shape contains non-integer values: {shape_str}")


# ============================================================================
# Export Function
# ============================================================================

def export_rfcnet_dcnv4_onnx(args: argparse.Namespace) -> None:
    """Export RFC Net with DCNv4 to ONNX."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    inner = build_model(args.weights)
    inner.to(device)

    # Wrapper for clean ONNX export
    class RFCNetExportWrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, masked_flows: torch.Tensor, masks: torch.Tensor):
            """
            Args:
                masked_flows: [B, T, 2, H, W] - masked optical flows
                masks: [B, T, 1, H, W] - binary masks
            Returns:
                flow: [B, T, 2, H, W] - completed flows
            """
            flow, _ = self.model(masked_flows, masks)
            return flow

    model = RFCNetExportWrapper(inner)
    model.eval()

    # Output path
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy inputs
    b, t, _, h, w = args.optshape or (1, 8, 2, 640, 480)
    print(f"\n[INFO] Export shape: B={b}, T={t}, H={h}, W={w}")

    dummy_masked_flows = torch.randn((b, t, 2, h, w), dtype=torch.float32, device=device)
    dummy_masks = torch.ones((b, t, 1, h, w), dtype=torch.float32, device=device)

    # Dynamic axes for TensorRT optimization profiles
    dynamic_axes = {
        "masked_flows": {0: "batch", 1: "time", 3: "height", 4: "width"},
        "masks": {0: "batch", 1: "time", 3: "height", 4: "width"},
        "flow": {0: "batch", 1: "time", 3: "height", 4: "width"},
    }

    print(f"\n[INFO] Exporting to {output_path}...")
    print(f"  Opset version: {args.opset}")
    print(f"  Dynamic axes: enabled")

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_masked_flows, dummy_masks),
        output_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["masked_flows", "masks"],
        output_names=["flow"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy exporter for better custom op support
        verbose=args.verbose,
    )

    print()
    print("=" * 80)
    print(f"[SUCCESS] RFC Net with DCNv4 exported to {output_path}")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Build DCNv4 TensorRT plugin:")
    print("     cd dcnv4_tensorrt_plugin && build.bat")
    print()
    print("  2. Build TensorRT engine:")
    print("     python build_rfcnet_trt_engine.py \\")
    print(f"         --onnx {output_path} \\")
    print("         --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine \\")
    print("         --fp16")
    print()

    if args.minshape or args.optshape or args.maxshape:
        print("  Optimization profiles:")
        if args.minshape:
            b, t, c, h, w = args.minshape
            print(f"     Min: {b}x{t}x{c}x{h}x{w}")
        if args.optshape:
            b, t, c, h, w = args.optshape
            print(f"     Opt: {b}x{t}x{c}x{h}x{w}")
        if args.maxshape:
            b, t, c, h, w = args.maxshape
            print(f"     Max: {b}x{t}x{c}x{h}x{w}")
        print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export RFC Net with DCNv4 to ONNX for TensorRT plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/recurrent_flow_completion.pth",
        help="Path to RFC Net weights",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="engines/rfcnet/rfcnet_dcnv4.onnx",
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )
    parser.add_argument(
        "--minshape",
        type=_parse_shape_5d,
        default=None,
        help="Minimum shape: BxTxCxHxW (e.g., 1x8x2x256x256)",
    )
    parser.add_argument(
        "--optshape",
        type=_parse_shape_5d,
        default=None,
        help="Optimal shape: BxTxCxHxW (e.g., 1x8x2x640x480)",
    )
    parser.add_argument(
        "--maxshape",
        type=_parse_shape_5d,
        default=None,
        help="Maximum shape: BxTxCxHxW (e.g., 1x16x2x1280x720)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ONNX export logging",
    )

    args = parser.parse_args()

    try:
        export_rfcnet_dcnv4_onnx(args)
    except Exception as e:
        print()
        print("=" * 80)
        print(f"[FAILED] EXPORT FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
