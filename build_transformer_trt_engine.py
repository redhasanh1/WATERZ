"""
Build TensorRT FP8 engine for Sparse Temporal Transformer

Compiles ONNX transformer to TensorRT FP8 optimized engine.
Target: 5-10x speedup (2.39s -> 0.24-0.48s per segment)
Uses RTX 4090 Ada Lovelace 4th Gen Tensor Cores (1,321 TFLOPs FP8)
"""

import sys
import os
import argparse
import time

# Setup TensorRT DLLs - add to PATH before importing
trt_lib = r"D:\watermarkz\TensorRT-10.13.3.9\lib"
trt_bin = r"D:\watermarkz\TensorRT-10.13.3.9\bin"
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

os.environ['PATH'] = f"{trt_lib};{trt_bin};{cuda_bin};{os.environ['PATH']}"

os.add_dll_directory(trt_lib)
os.add_dll_directory(trt_bin)
os.add_dll_directory(cuda_bin)

import tensorrt as trt
import numpy as np
import ctypes

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_col2im_plugin():
    """
    Load Col2Im TensorRT plugin DLL

    This plugin is required for FP8 TensorRT transformer build.
    The ONNX model contains Col2Im operations that TensorRT doesn't natively support.

    Returns:
        bool: True if plugin loaded successfully, False otherwise
    """
    # Check both possible DLL locations
    plugin_paths = [
        r"D:\watermarkz\col2im_tensorrt_plugin\build\Release\col2im_plugin.dll",  # MSVC Release config
        r"D:\watermarkz\col2im_tensorrt_plugin\build\col2im_plugin.dll"  # Direct build
    ]

    plugin_path = None
    for path in plugin_paths:
        if os.path.exists(path):
            plugin_path = path
            break

    if not plugin_path:
        print()
        print("=" * 80)
        print("[WARNING] Col2Im plugin DLL not found!")
        print("=" * 80)
        print("[PATH] Checked locations:")
        for path in plugin_paths:
            print(f"        {path}")
        print()
        print("[INFO] The transformer ONNX contains Col2Im operations that require")
        print("[INFO] a custom TensorRT plugin. Without this plugin, the build will FAIL.")
        print()
        print("[BUILD] To build the plugin:")
        print("        1. Open Windows Command Prompt or PowerShell")
        print("        2. Run: python build_col2im_ninja.py")
        print("        3. Wait 2-5 minutes for compilation")
        print("        4. Re-run this engine build")
        print()
        print("[ALTERNATIVE] See BUILD_COL2IM_INSTRUCTIONS.md for detailed steps")
        print("=" * 80)
        print()
        return False

    print()
    print("[PLUGIN] Loading Col2Im TensorRT plugin...")
    try:
        # Load plugin DLL
        plugin_lib = ctypes.CDLL(plugin_path)
        print(f"[OK] Col2Im plugin DLL loaded: {plugin_path}")

        # Initialize TensorRT plugin registry
        # This registers all plugins in the DLL with TensorRT
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        print("[OK] TensorRT plugin registry initialized")

        # Get plugin registry to verify Col2Im is registered
        plugin_registry = trt.get_plugin_registry()
        plugin_creators = plugin_registry.plugin_creator_list

        col2im_found = False
        for creator in plugin_creators:
            if "Col2Im" in creator.name:
                print(f"[OK] Col2Im plugin registered: {creator.name} v{creator.plugin_version}")
                col2im_found = True
                break

        if not col2im_found:
            print("[WARNING] Col2Im plugin not found in registry")
            print("[WARNING] This may cause build failure if ONNX contains Col2Im ops")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to load Col2Im plugin: {e}")
        print(f"[ERROR] DLL path: {plugin_path}")
        print()
        print("[INFO] Possible causes:")
        print("        - DLL incompatible with TensorRT 10.13.3.9")
        print("        - Missing CUDA 12.6 runtime DLLs")
        print("        - Plugin compilation errors")
        print()
        print("[INFO] Try rebuilding the plugin with BUILD_COL2IM_PLUGIN.bat")
        return False


def build_engine(onnx_path, engine_path, fp16=True, fp8=False, int8=False, workspace_size=8, verbose=False):
    """
    Build TensorRT engine from ONNX model

    Args:
        onnx_path: Path to input ONNX file
        engine_path: Path to output engine file
        fp16: Enable FP16 precision (default True)
        fp8: Enable FP8 precision (requires Hopper H100, not RTX 4090!)
        int8: Enable INT8 precision (RTX 4090 native: 1,321 TOPS)
        workspace_size: Max workspace size in GB (default 8)
        verbose: Verbose logging
    """

    if verbose:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

    print(f"[BUILD] Loading ONNX model: {onnx_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    # Load Col2Im TensorRT plugin (required for transformer ONNX)
    plugin_loaded = load_col2im_plugin()
    if not plugin_loaded:
        print()
        print("[WARNING] Continuing without Col2Im plugin...")
        print("[WARNING] Build will likely FAIL if ONNX contains Col2Im operations")
        print()

    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("[BUILD] Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX file")
            for i in range(parser.num_errors):
                print(f"  Error {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    print(f"[OK] ONNX parsed successfully")
    print(f"[INFO] Network inputs: {network.num_inputs}")
    print(f"[INFO] Network outputs: {network.num_outputs}")

    # Print network info
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"[INFO] Input {i}: {input_tensor.name}, shape={input_tensor.shape}, dtype={input_tensor.dtype}")

    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"[INFO] Output {i}: {output_tensor.name}, shape={output_tensor.shape}, dtype={output_tensor.dtype}")

    # Create builder config
    config = builder.create_builder_config()

    # Set workspace size
    workspace_bytes = workspace_size * (1 << 30)  # GB to bytes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    print(f"[CONFIG] Workspace size: {workspace_size} GB")

    # Enable FP16 precision
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[CONFIG] FP16 precision enabled (Ada Lovelace Tensor Cores)")
        else:
            print("[WARNING] FP16 not supported on this platform")

    # Enable FP8 precision (Hopper H100 only - NOT RTX 4090!)
    if fp8:
        # TensorRT 10.13.3.9 has BuilderFlag.FP8 but RTX 4090 lacks FP8 tensor cores
        # This will FAIL or provide no speedup on RTX 4090 (CC 8.9)
        try:
            config.set_flag(trt.BuilderFlag.FP8)
            print("[CONFIG] FP8 precision flag set (WARNING: RTX 4090 has NO FP8 tensor cores!)")
            print("[CONFIG] Expected: No speedup, storage-only FP8")
            print("[CONFIG] Recommendation: Use INT8 instead for RTX 4090")
        except Exception as e:
            print(f"[WARNING] FP8 flag failed: {e}")

    # Check if input has dynamic dimensions
    input_tensor = network.get_input(0)
    has_dynamic_shape = any(dim == -1 for dim in input_tensor.shape)

    if has_dynamic_shape:
        # Set optimization profile for dynamic shapes (must be done BEFORE INT8 calibrator setup)
        print("[BUILD] Setting optimization profile for dynamic T, H, W dimensions...")
        profile = builder.create_optimization_profile()

        # Input shape: [B, T, H, W, C] = [1, T, H, W, 512]
        # T is dynamic (varies from 5 to 20 frames per segment based on neighbor frames)
        # H is dynamic (varies from 15 to 25 tokens based on video height)
        # W is dynamic (varies from 10 to 40 tokens based on video width)
        input_name = network.get_input(0).name
        profile.set_shape(
            input_name,
            min=(1, 5, 15, 10, 512),     # Min: 5 frames, small video (boundary case)
            opt=(1, 12, 20, 36, 512),    # Optimal: 12 frames, 480x872 video (typical)
            max=(1, 20, 25, 40, 512)     # Max: 20 frames, large video (with ref frames)
        )
        config.add_optimization_profile(profile)
        print("[OK] Optimization profile set (T: 5-20, H: 15-25, W: 10-40)")
    else:
        print("[INFO] Input has static shape, skipping optimization profile")

    # Setup INT8 calibrator (must be done AFTER optimization profile)
    if int8:
        if builder.platform_has_fast_int8:
            # Load calibrator
            from transformer_int8_calibrator import TransformerEntropyCalibrator

            calibration_dir = r"C:\Users\has\Documents\watermarkzc\calibration_data\transformer"
            cache_file = "engines/transformer/transformer_int8_calibration.cache"

            print("[CALIBRATOR] INT8 precision enabled (RTX 4090 native: 1,321 TOPS)")
            print(f"[CALIBRATOR] Calibration dir: {calibration_dir}")
            print("[CALIBRATOR] Using EntropyCalibrator2 (100 samples)")

            try:
                calibrator = TransformerEntropyCalibrator(
                    calibration_dir=calibration_dir,
                    cache_file=cache_file,
                    batch_size=1
                )
                if has_dynamic_shape:
                    config.set_calibration_profile(profile)
                config.int8_calibrator = calibrator
                print("[CALIBRATOR] Calibrator configured")
                print("[CALIBRATOR] Expected: 1.5-2.0x speedup over FP16")
            except Exception as e:
                print(f"[ERROR] Failed to setup calibrator: {e}")
                raise

    # Build engine
    print("[BUILD] Building TensorRT engine...")
    print("[BUILD] This may take 5-15 minutes (kernel auto-tuning)...")
    if int8:
        print("[BUILD] INT8 calibration will run during build (1-2 min)...")
    start_time = time.time()

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    build_time = time.time() - start_time
    print(f"[OK] Engine built in {build_time:.1f} seconds")

    # Save engine
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    # Get file size from saved file
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"[OK] Engine saved: {engine_path} ({engine_size_mb:.1f} MB)")

    # Verify engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")

    print("[OK] Engine verification passed")
    # Note: max_batch_size, num_bindings, has_implicit_batch_dimension removed in TensorRT 10.x

    return engine_path


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT engine for Sparse Transformer')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Input ONNX file path')
    parser.add_argument('--engine', type=str, required=True,
                        help='Output TensorRT engine file path')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 precision (default: True)')
    parser.add_argument('--fp8', action='store_true', default=False,
                        help='Enable FP8 precision (Hopper H100 only, NOT RTX 4090!)')
    parser.add_argument('--int8', action='store_true', default=False,
                        help='Enable INT8 precision (RTX 4090: 1,321 TOPS INT8)')
    parser.add_argument('--workspace', type=int, default=8,
                        help='Max workspace size in GB (default: 8)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose TensorRT logging')

    args = parser.parse_args()

    print("=" * 80)
    print("TENSORRT SPARSE TRANSFORMER ENGINE BUILD")
    print("=" * 80)
    print(f"Input ONNX: {args.onnx}")
    print(f"Output engine: {args.engine}")
    print(f"FP16: {args.fp16}")
    print(f"FP8: {args.fp8} (WARNING: RTX 4090 has NO FP8 tensor cores!)")
    print(f"INT8: {args.int8} (RTX 4090 native: 1,321 TOPS)")
    print(f"Workspace: {args.workspace} GB")
    print(f"Target: 1.5-2.0x speedup with INT8 (547ms -> 273-365ms)")
    print("=" * 80)
    print()

    try:
        engine_path = build_engine(
            onnx_path=args.onnx,
            engine_path=args.engine,
            fp16=args.fp16,
            fp8=args.fp8,
            int8=args.int8,
            workspace_size=args.workspace,
            verbose=args.verbose
        )

        print()
        print("=" * 80)
        print("[SUCCESS] TensorRT engine build complete!")
        print(f"[ENGINE] {engine_path}")
        print()
        print("[NEXT] Test the engine:")
        print(f"       python test_transformer_trt.py --engine {engine_path}")
        print()
        print("[NEXT] Integrate into production:")
        print(f"       1. Set FORCE_TRT_TRANSFORMER=1 in START_CELERY_TRT.bat")
        print(f"       2. Restart Celery workers")
        print(f"       3. Expected: 2.39s -> 0.24-0.48s per segment (5-10x speedup!)")
        print("=" * 80)

    except Exception as e:
        print()
        print("=" * 80)
        print(f"[ERROR] Engine build failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
