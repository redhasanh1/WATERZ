"""
TensorRT Engine Builder for RFC Net with DCNv4 Plugin

Builds optimized FP16 TensorRT engine for RFC Net with custom DCNv4 plugin.
Target: 6.5-9x speedup vs PyTorch baseline

Usage:
    python build_rfcnet_trt_engine.py \\
        --onnx engines/rfcnet/rfcnet_dcnv4.onnx \\
        --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine \\
        --fp16
"""
import argparse
import os
import sys
import ctypes
from pathlib import Path

# Setup TensorRT and CUDA paths BEFORE importing tensorrt
SCRIPT_DIR = Path(__file__).parent.absolute()
TENSORRT_LIB = SCRIPT_DIR / "TensorRT-10.13.3.9" / "lib"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

# CRITICAL: Update PATH first (TensorRT's import searches PATH)
os.environ['PATH'] = str(TENSORRT_LIB) + os.pathsep + CUDA_BIN + os.pathsep + os.environ.get('PATH', '')

# Windows Python 3.8+ also requires os.add_dll_directory for ctypes loading
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    os.add_dll_directory(str(TENSORRT_LIB))
    os.add_dll_directory(CUDA_BIN)

import tensorrt as trt

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_ONNX = "engines/rfcnet/rfcnet_dcnv4.onnx"
DEFAULT_ENGINE = "engines/rfcnet/rfcnet_dcnv4_fp16.engine"
DEFAULT_PLUGIN_PATH = "dcnv4_tensorrt_plugin/build/Release/dcnv4_plugin.dll"


# ============================================================================
# TensorRT Engine Builder
# ============================================================================

def load_dcnv4_plugin(plugin_path: str, logger: trt.Logger) -> bool:
    """
    Load DCNv4 TensorRT plugin.

    Args:
        plugin_path: Path to dcnv4_plugin.dll (Windows) or libdcnv4_plugin.so (Linux)
        logger: TensorRT logger

    Returns:
        True if plugin loaded successfully
    """
    print("[*] Loading DCNv4 TensorRT plugin...")
    print(f"   Plugin path: {plugin_path}")

    # Check if plugin file exists
    plugin_file = Path(plugin_path)
    if not plugin_file.exists():
        print(f"[!] Plugin file not found: {plugin_path}")
        print()
        print("Please build the plugin first:")
        print("   cd dcnv4_tensorrt_plugin")
        print("   build.bat")
        return False

    # Add plugin directory to DLL search path (Windows)
    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        plugin_dir = plugin_file.parent.absolute()
        os.add_dll_directory(str(plugin_dir))
        print(f"   Added DLL directory: {plugin_dir}")

    # Load plugin library
    try:
        ctypes.CDLL(str(plugin_file))
        print(f"[+] Plugin library loaded: {plugin_path}")
    except Exception as e:
        print(f"[!] Failed to load plugin: {e}")
        print(f"   Make sure TensorRT and CUDA DLLs are accessible")
        return False

    # Initialize TensorRT plugins
    trt.init_libnvinfer_plugins(logger, '')
    print("[+] TensorRT plugins initialized")

    # Verify DCNv4 plugin is registered
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("DCNv4Plugin", "1", "")

    if creator is None:
        print("[!] DCNv4Plugin not found in registry!")
        print()
        print("Available plugins:")
        plugin_list = [registry.get_plugin_creator(i) for i in range(registry.num_plugin_creators)]
        for p in plugin_list:
            if p is not None:
                print(f"   - {p.name} v{p.plugin_version}")
        return False

    print(f"[+] DCNv4Plugin registered: {creator.name} v{creator.plugin_version}")
    print()
    return True


def build_engine(args: argparse.Namespace) -> bool:
    """
    Build TensorRT engine for RFC Net with DCNv4.

    Args:
        args: Command-line arguments

    Returns:
        True if engine built successfully
    """
    print("=" * 80)
    print("RFC NET TENSORRT ENGINE BUILDER (WITH DCNv4 PLUGIN)")
    print("=" * 80)
    print()

    # TensorRT Logger
    log_level = trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO
    TRT_LOGGER = trt.Logger(log_level)

    # Step 1: Load DCNv4 plugin
    if not load_dcnv4_plugin(args.plugin, TRT_LOGGER):
        return False

    # Step 2: Create builder
    print("[*] Creating TensorRT builder...")
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    print("[+] Builder created")
    print()

    # Step 3: Parse ONNX model
    print("[*] Parsing ONNX model...")
    print(f"   ONNX path: {args.onnx}")

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx, 'rb') as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        print("[!] ONNX parsing failed!")
        print()
        print("Errors:")
        for i in range(parser.num_errors):
            error = parser.get_error(i)
            print(f"   [{i}] {error}")
        print()
        print("This usually means:")
        print("  - ONNX model contains unsupported ops")
        print("  - DCNv4 custom op not properly registered")
        print("  - ONNX export used wrong opset version")
        return False

    print(f"[+] ONNX parsed successfully")
    print(f"   Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"     [{i}] {inp.name}: {inp.shape}")

    print(f"   Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"     [{i}] {out.name}: {out.shape}")

    print(f"   Network layers: {network.num_layers}")
    print()

    # Step 4: Configure optimization
    print("[*] Configuring TensorRT optimization...")

    # FP16 precision (1.3-1.5x speedup on RTX 4090)
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("   [+] FP16 enabled (1.5-2x speedup)")
    else:
        print("   [+] FP32 (baseline precision)")

    # Workspace size (for complex ops and plugin workspace)
    workspace_gb = args.workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    print(f"   [+] Workspace: {workspace_gb}GB")

    # Tactic sources
    config.set_tactic_sources(
        1 << int(trt.TacticSource.CUBLAS) |
        1 << int(trt.TacticSource.CUBLAS_LT) |
        1 << int(trt.TacticSource.CUDNN)
    )
    print("   [+] All tactic sources enabled")

    # Builder optimization level
    config.builder_optimization_level = 5  # Maximum
    print("   [+] Optimization level: 5 (maximum)")

    # Step 5: Configure dynamic shapes
    print()
    print("[*] Configuring dynamic shape profiles...")

    profile = builder.create_optimization_profile()

    # Parse shape strings (BxTxCxHxW format)
    def parse_shape(shape_str):
        parts = shape_str.split('x')
        if len(parts) != 5:
            raise ValueError(f"Shape must be BxTxCxHxW, got: {shape_str}")
        return tuple(int(p) for p in parts)

    min_shape = parse_shape(args.min_shape)
    opt_shape = parse_shape(args.opt_shape)
    max_shape = parse_shape(args.max_shape)

    # masked_flows input: [B, T, 2, H, W]
    profile.set_shape(
        "masked_flows",
        (min_shape[0], min_shape[1], 2, min_shape[3], min_shape[4]),  # min
        (opt_shape[0], opt_shape[1], 2, opt_shape[3], opt_shape[4]),  # opt
        (max_shape[0], max_shape[1], 2, max_shape[3], max_shape[4])   # max
    )

    # masks input: [B, T, 1, H, W]
    profile.set_shape(
        "masks",
        (min_shape[0], min_shape[1], 1, min_shape[3], min_shape[4]),  # min
        (opt_shape[0], opt_shape[1], 1, opt_shape[3], opt_shape[4]),  # opt
        (max_shape[0], max_shape[1], 1, max_shape[3], max_shape[4])   # max
    )

    config.add_optimization_profile(profile)

    print(f"   [+] Optimization profile configured:")
    print(f"       Min: {args.min_shape}")
    print(f"       Opt: {args.opt_shape}")
    print(f"       Max: {args.max_shape}")
    print()

    # Step 6: Build engine
    print("[*] Building TensorRT engine...")
    print("   This may take 5-10 minutes...")
    print("   TensorRT will test thousands of kernel configurations...")
    print()

    serialized_engine = builder.build_serialized_network(network, config)

    if not serialized_engine:
        print("[!] Engine build failed!")
        print()
        print("Common causes:")
        print("  - Unsupported layer configuration")
        print("  - Plugin incompatibility")
        print("  - Insufficient memory")
        return False

    # Step 7: Save engine
    output_path = Path(args.engine)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert IHostMemory to bytes for writing
    engine_bytes = bytes(serialized_engine)

    with open(output_path, 'wb') as f:
        f.write(engine_bytes)

    engine_size_mb = len(engine_bytes) / (1024 * 1024)

    print()
    print("=" * 80)
    print("[SUCCESS] TensorRT engine built!")
    print("=" * 80)
    print()
    print(f"   Engine file: {output_path}")
    print(f"   Engine size: {engine_size_mb:.1f} MB")
    print(f"   Precision: {'FP16' if args.fp16 else 'FP32'}")
    print(f"   Optimization profiles:")
    print(f"     Min: {args.min_shape}")
    print(f"     Opt: {args.opt_shape}")
    print(f"     Max: {args.max_shape}")
    print()
    print("Next steps:")
    print("  1. Test inference:")
    print(f"     python test_rfcnet_trt_dcnv4.py --engine {output_path}")
    print()
    print("  2. Benchmark performance:")
    print("     python benchmark_rfcnet_trt.py")
    print()

    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine for RFC Net with DCNv4 plugin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--onnx",
        type=str,
        default=DEFAULT_ONNX,
        help=f"Input ONNX model path (default: {DEFAULT_ONNX})",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=DEFAULT_ENGINE,
        help=f"Output TensorRT engine path (default: {DEFAULT_ENGINE})",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default=DEFAULT_PLUGIN_PATH,
        help=f"DCNv4 plugin path (default: {DEFAULT_PLUGIN_PATH})",
    )

    # Precision
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision (recommended, 1.5-2x speedup)",
    )

    # Dynamic shapes (BxTxCxHxW format)
    parser.add_argument(
        "--min-shape",
        type=str,
        default="1x8x2x256x256",
        help="Minimum input shape: BxTxCxHxW (default: 1x8x2x256x256)",
    )
    parser.add_argument(
        "--opt-shape",
        type=str,
        default="1x8x2x640x480",
        help="Optimal input shape: BxTxCxHxW (default: 1x8x2x640x480)",
    )
    parser.add_argument(
        "--max-shape",
        type=str,
        default="1x16x2x1280x720",
        help="Maximum input shape: BxTxCxHxW (default: 1x16x2x1280x720)",
    )

    # Configuration
    parser.add_argument(
        "--workspace",
        type=int,
        default=4,
        help="Workspace size in GB (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose TensorRT logging",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.onnx).exists():
        print(f"[!] ONNX file not found: {args.onnx}")
        print()
        print("Please export ONNX model first:")
        print("   python export_rfcnet_dcnv4_onnx.py")
        sys.exit(1)

    try:
        success = build_engine(args)
        sys.exit(0 if success else 1)
    except Exception as e:
        print()
        print("=" * 80)
        print(f"[!] FATAL ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
