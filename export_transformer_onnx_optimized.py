"""
Export Sparse Temporal Transformer to ONNX for TensorRT
Optimized export with static shapes and graph optimizations

Expected TensorRT FP16 speedup: 2-5x over PyTorch FP8
Target: 99ms -> 20-50ms per inference (2-5x faster!)
"""
import torch
import torch.nn as nn
import os
import sys

# Add faster-propainter to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'faster-propainter-main'))

from model.modules.sparse_transformer import TemporalSparseTransformerBlock


class TransformerONNXWrapper(nn.Module):
    """
    Wrapper for exporting TemporalSparseTransformerBlock to ONNX

    Handles:
    - Static shape enforcement
    - Input/output simplification
    - Removing dynamic ops (if any)
    """
    def __init__(self, transformer, static_shape):
        super().__init__()
        self.transformer = transformer
        self.B, self.T, self.H, self.W, self.C = static_shape

    def forward(self, x):
        """
        Simplified forward for ONNX export

        Args:
            x: [B, T, H, W, C] tensor (static shape)

        Returns:
            output: [B, T, H, W, C] tensor
        """
        # Static fold_x_size (known at export time)
        fold_x_size = (self.H, self.W)

        # Dummy mask for ONNX export (all ones = no masking)
        # Shape: [B, T, H, W, 1]
        l_mask = torch.ones(self.B, self.T, self.H, self.W, 1,
                           device=x.device, dtype=x.dtype)

        # Fixed t_dilation
        t_dilation = 2

        # Forward pass
        output = self.transformer(x, fold_x_size, l_mask, t_dilation)

        return output


def export_transformer_to_onnx(
    checkpoint_path=None,
    output_path="transformer_fp16.onnx",
    static_shape=(1, 5, 15, 15, 512),  # After token merging: 15x15
    opset_version=17,
    simplify=True
):
    """
    Export transformer to ONNX format optimized for TensorRT

    Args:
        checkpoint_path: Path to transformer checkpoint (or None for random weights)
        output_path: Output ONNX file path
        static_shape: (B, T, H, W, C) - must match production inference!
        opset_version: ONNX opset version (17 for best TensorRT compatibility)
        simplify: Apply ONNX simplifier (recommended)

    Returns:
        ONNX model path
    """
    print("="*60)
    print("TRANSFORMER ONNX EXPORT")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create transformer
    B, T, H, W, C = static_shape

    print(f"\n1. Creating transformer...")
    print(f"   Input shape: {static_shape}")
    print(f"   Device: {device}")

    # t2t_params for token-to-token operations (required by FusionFeedForward)
    t2t_params = {
        'kernel_size': (7, 7),
        'stride': (3, 3),
        'padding': (3, 3)
    }

    transformer = TemporalSparseTransformerBlock(
        dim=C,
        n_head=4,
        window_size=(3, 8, 8),
        pool_size=(4, 4),
        depths=8,  # 8 layers
        t2t_params=t2t_params
    ).to(device)

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n2. Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        transformer.load_state_dict(checkpoint, strict=False)
    else:
        print(f"\n2. Using random weights (no checkpoint provided)")

    transformer.eval()

    # Disable optimizations that might not export well
    # We want clean ONNX for TensorRT to optimize
    os.environ['ENABLE_FLASH_ATTENTION'] = '0'  # Export with standard attention
    os.environ['ENABLE_FP8_TRANSFORMER'] = '0'  # Export FP32, convert to FP16 in TensorRT
    os.environ['ENABLE_TOKEN_MERGING'] = '0'    # Export full transformer (apply token merging separately)

    # Create wrapper
    wrapper = TransformerONNXWrapper(transformer, static_shape).to(device)
    wrapper.eval()

    # Create dummy input
    print(f"\n3. Creating dummy input...")
    dummy_input = torch.randn(*static_shape, device=device)

    # Test forward pass
    print(f"\n4. Testing forward pass...")
    with torch.no_grad():
        output = wrapper(dummy_input)
    print(f"   Output shape: {output.shape}")

    # Export to ONNX
    print(f"\n5. Exporting to ONNX...")
    print(f"   Output: {output_path}")
    print(f"   Opset: {opset_version}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,  # Static shapes for TensorRT optimization
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )

    print(f"   ✓ ONNX exported: {output_path}")

    # Simplify ONNX (optional but recommended)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print(f"\n6. Simplifying ONNX...")
            model = onnx.load(output_path)
            model_simp, check = onnx_simplify(model)

            if check:
                onnx.save(model_simp, output_path)
                print(f"   ✓ ONNX simplified")
            else:
                print(f"   ⚠ Simplification failed (using original)")

        except ImportError:
            print(f"\n6. ONNX simplifier not available (pip install onnx-simplifier)")

    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Build TensorRT engine:")
    print(f"   trtexec --onnx={output_path} \\")
    print(f"           --saveEngine=transformer_fp16.trt \\")
    print(f"           --fp16 \\")
    print(f"           --workspace=4096 \\")
    print(f"           --verbose")
    print(f"\n2. Expected speedup: 2-5x (99ms -> 20-50ms)")
    print(f"3. Production integration: Load .trt engine with TensorRT Python API")
    print("="*60)

    return output_path


def build_tensorrt_engine(
    onnx_path="transformer_fp16.onnx",
    engine_path="transformer_fp16.trt",
    fp16=True,
    workspace_mb=4096,
    verbose=True
):
    """
    Build TensorRT engine from ONNX model

    Args:
        onnx_path: Input ONNX file
        engine_path: Output TensorRT engine file
        fp16: Enable FP16 precision
        workspace_mb: TensorRT workspace size in MB
        verbose: Print verbose logs

    Returns:
        TensorRT engine path
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[ERROR] TensorRT not available (pip install tensorrt)")
        return None

    print("="*60)
    print("TENSORRT ENGINE BUILD")
    print("="*60)

    # Create logger
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    # Create builder
    print(f"\n1. Creating TensorRT builder...")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"\n2. Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX:")
            for error in range(parser.num_errors):
                print(f"  - {parser.get_error(error)}")
            return None

    print(f"   ✓ ONNX parsed successfully")

    # Configure builder
    print(f"\n3. Configuring TensorRT builder...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"   ✓ FP16 mode enabled")

    print(f"   Workspace: {workspace_mb} MB")

    # Build engine
    print(f"\n4. Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("[ERROR] Failed to build TensorRT engine")
        return None

    # Save engine
    print(f"\n5. Saving engine: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"   ✓ Engine saved: {engine_path}")

    print("\n" + "="*60)
    print("ENGINE BUILD COMPLETE!")
    print("="*60)

    return engine_path


def benchmark_tensorrt_vs_pytorch():
    """
    Benchmark TensorRT engine vs PyTorch baseline
    """
    print("TensorRT benchmark coming soon...")
    print("First export ONNX, then build TRT engine, then benchmark")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export transformer to ONNX/TensorRT")
    parser.add_argument('--checkpoint', type=str, default=None, help='Transformer checkpoint path')
    parser.add_argument('--output', type=str, default='transformer_fp16.onnx', help='Output ONNX path')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--temporal', type=int, default=5, help='Temporal length')
    parser.add_argument('--height', type=int, default=15, help='Height (after token merging)')
    parser.add_argument('--width', type=int, default=15, help='Width (after token merging)')
    parser.add_argument('--channels', type=int, default=512, help='Channel dimension')
    parser.add_argument('--build-trt', action='store_true', help='Also build TensorRT engine')
    parser.add_argument('--trt-engine', type=str, default='transformer_fp16.trt', help='TensorRT engine path')

    args = parser.parse_args()

    # Export to ONNX
    static_shape = (args.batch, args.temporal, args.height, args.width, args.channels)

    onnx_path = export_transformer_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        static_shape=static_shape,
        opset_version=17,
        simplify=True
    )

    # Build TensorRT engine (if requested)
    if args.build_trt:
        build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=args.trt_engine,
            fp16=True,
            workspace_mb=4096,
            verbose=True
        )
