"""
Export Sparse Temporal Transformer to ONNX for TensorRT compilation

This script exports the TemporalSparseTransformerBlock (8 layers) to ONNX format.
Target: 5-10x speedup with TensorRT FP8 (2.39s → 0.24-0.48s)
"""

import sys
import os
import torch
import torch.onnx
import argparse

# Add faster-propainter-main to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'faster-propainter-main'))

from model.modules.sparse_transformer import TemporalSparseTransformerBlock


class TransformerWrapper(torch.nn.Module):
    """Wrapper to make transformer ONNX-exportable

    SIMPLIFIED VERSION: Explicitly handles batch=1 to avoid TensorRT reshape issues.
    Removes problematic dynamic shape patterns that cause TensorRT errors.
    """
    def __init__(self, dim=512, n_head=4, window_size=(5, 9), pool_size=(4, 4), depths=8):
        super().__init__()

        # T2T params (required by FusionFeedForward, same as ProPainter)
        t2t_params = {
            'kernel_size': (7, 7),
            'stride': (3, 3),
            'padding': (3, 3)
        }

        self.transformer = TemporalSparseTransformerBlock(
            dim=dim,
            n_head=n_head,
            window_size=window_size,
            pool_size=pool_size,
            depths=depths,
            t2t_params=t2t_params
        )
        self.dim = dim
        self.depths = depths

        # Store default fold_x_size (typical for 480x872 input after 8x downsampling)
        # Original: 480x872 → Downsampled: 60x109 (but code uses 60x108)
        # NOTE: fold_x_size will be computed dynamically based on actual H/W
        self.fold_h_default = 60
        self.fold_w_default = 108

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W, C] - feature tokens
               B=1 (FIXED), T=variable frames, H=DYNAMIC (10-25), W=DYNAMIC (10-40), C=512 (FIXED)

        Returns:
            out: [B, T, H, W, C] - transformed features

        DYNAMIC SHAPES: H and W now vary based on crop region size
        fold_x_size computed dynamically: fold_x_size = (H * 3, W * 3)
        """
        B, T, H, W, C = x.shape

        # Verify fixed dimensions only
        assert B == 1, f"Batch size must be 1, got {B}"
        assert C == self.dim, f"Channels must be {self.dim}, got {C}"

        # Compute fold_x_size dynamically based on H/W
        # fold_x_size = (H * 3, W * 3) matches ProPainter's downsampling ratio
        fold_h = H * 3
        fold_w = W * 3
        fold_x_size = (fold_h, fold_w)

        # Create dummy mask (all ones = no masking)
        # Shape: [1, T, H, W, 1] - H/W now dynamic
        l_mask = torch.ones(1, T, H, W, 1, device=x.device, dtype=x.dtype)

        # Use default t_dilation=2
        out = self.transformer(x, fold_x_size, l_mask, t_dilation=2)

        # Output shape should match input
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

        return out


def export_transformer_onnx(output_path, opset_version=17, verbose=True):
    """Export transformer to ONNX with dynamic temporal axis"""

    print("[EXPORT] Creating Sparse Temporal Transformer (8 layers, 512 dim, 4 heads)...")

    # Create transformer (same config as ProPainter checkpoint)
    model = TransformerWrapper(
        dim=512,
        n_head=4,
        window_size=(5, 9),
        pool_size=(4, 4),
        depths=8
    )
    model.eval()
    model.cuda()

    print(f"[EXPORT] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create dummy input
    # B=1, T=10 (variable), H=20, W=36, C=512
    # This represents ~10 frames at 480x872 resolution after encoder downsampling
    B, T, H, W, C = 1, 10, 20, 36, 512
    dummy_input = torch.randn(B, T, H, W, C, dtype=torch.float32).cuda()

    print(f"[EXPORT] Dummy input shape: {dummy_input.shape}")
    print(f"[EXPORT] Running forward pass to verify...")

    with torch.no_grad():
        dummy_output = model(dummy_input)
        print(f"[EXPORT] Output shape: {dummy_output.shape}")
        assert dummy_output.shape == dummy_input.shape, "Shape mismatch!"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f"[EXPORT] Exporting to ONNX (opset {opset_version})...")
    print(f"[EXPORT] Output: {output_path}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {1: 'temporal', 2: 'height', 3: 'width'},   # T, H, W all dynamic (B always = 1)
            'output': {1: 'temporal', 2: 'height', 3: 'width'}
        },
        verbose=verbose
    )

    print(f"[OK] ONNX export complete: {output_path}")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model verification passed")

        # Print model info
        print(f"[INFO] ONNX IR version: {onnx_model.ir_version}")
        print(f"[INFO] ONNX opset version: {onnx_model.opset_import[0].version}")
        print(f"[INFO] Input shape: {onnx_model.graph.input[0].type.tensor_type.shape.dim}")
        print(f"[INFO] Output shape: {onnx_model.graph.output[0].type.tensor_type.shape.dim}")

    except ImportError:
        print("[WARNING] onnx package not found, skipping verification")
    except Exception as e:
        print(f"[WARNING] ONNX verification failed: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export Sparse Transformer to ONNX')
    parser.add_argument('--out', type=str, default='engines/transformer/transformer.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=19,
                        help='ONNX opset version (default: 19, required for FP8 Q/DQ nodes)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose ONNX export logging')

    args = parser.parse_args()

    print("=" * 80)
    print("SPARSE TEMPORAL TRANSFORMER ONNX EXPORT")
    print("=" * 80)
    print(f"Output: {args.out}")
    print(f"ONNX opset: {args.opset}")
    print(f"Target: TensorRT FP8 compilation (5-10x speedup)")
    print(f"Expected: 2.39s -> 0.24-0.48s per segment")
    print("=" * 80)
    print()

    try:
        export_transformer_onnx(
            output_path=args.out,
            opset_version=args.opset,
            verbose=args.verbose
        )

        print()
        print("=" * 80)
        print("[SUCCESS] ONNX export complete!")
        print(f"[NEXT] Build TensorRT engine:")
        print(f"       python build_transformer_trt_engine.py --onnx {args.out}")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print(f"[ERROR] Export failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
