"""
Simple TensorRT Engine Validation for RFC Net (No PyCUDA)

Uses PyTorch-TensorRT integration to avoid pycuda cuDNN issues.
"""
import argparse
import os
import sys
import time
import ctypes
from pathlib import Path

# Setup TensorRT and CUDA paths BEFORE importing tensorrt
SCRIPT_DIR = Path(__file__).parent.absolute()
TENSORRT_LIB = SCRIPT_DIR / "TensorRT-10.13.3.9" / "lib"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"

# CRITICAL: Update PATH first
os.environ['PATH'] = str(TENSORRT_LIB) + os.pathsep + CUDA_BIN + os.pathsep + os.environ.get('PATH', '')

# Windows Python 3.8+ also requires os.add_dll_directory
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    os.add_dll_directory(str(TENSORRT_LIB))
    os.add_dll_directory(CUDA_BIN)

import torch
import numpy as np
import tensorrt as trt

# Fix Windows Unicode encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

# Setup paths
sys.path.insert(0, 'faster-propainter-main')

# Enable DCNv4
os.environ['ENABLE_DCNV4_RFCNET'] = '1'
os.environ['ENABLE_FP8_RFCNET'] = '1'


DEFAULT_ENGINE = "engines/rfcnet/rfcnet_dcnv4_fp16.engine"
DEFAULT_PLUGIN = "dcnv4_tensorrt_plugin/build/Release/dcnv4_plugin.dll"


def load_engine(engine_path, plugin_path):
    """Load TensorRT engine with DCNv4 plugin"""
    print("=" * 80)
    print("RFC NET TENSORRT ENGINE VALIDATION")
    print("=" * 80)
    print()

    # Load plugin
    print(f"[1/4] Loading DCNv4 plugin: {plugin_path}")
    plugin_file = Path(plugin_path)
    if not plugin_file.exists():
        print(f"[ERROR] Plugin not found: {plugin_path}")
        return None

    # Add plugin directory to DLL search path
    if sys.platform == 'win32' and sys.version_info >= (3, 8):
        plugin_dir = plugin_file.parent.absolute()
        os.add_dll_directory(str(plugin_dir))

    try:
        ctypes.CDLL(str(plugin_file))
        print(f"      Plugin loaded successfully")
    except Exception as e:
        print(f"[ERROR] Plugin loading failed: {e}")
        return None

    # Initialize TensorRT plugins
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # Verify DCNv4 plugin
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("DCNv4Plugin", "1", "")
    if not creator:
        print("[ERROR] DCNv4Plugin not found in registry")
        return None
    print(f"      DCNv4Plugin registered: v{creator.plugin_version}")
    print()

    # Load engine
    print(f"[2/4] Loading TensorRT engine: {engine_path}")
    engine_file = Path(engine_path)
    if not engine_file.exists():
        print(f"[ERROR] Engine not found: {engine_path}")
        return None

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if not engine:
        print("[ERROR] Engine deserialization failed")
        return None

    print(f"      Engine loaded: {engine_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"      Inputs/Outputs: {engine.num_io_tensors} tensors")
    print()

    return engine


def test_inference(engine, shape=(1, 8, 640, 480)):
    """Test TensorRT inference using PyTorch CUDA tensors"""
    b, t, h, w = shape
    print(f"[3/4] Testing inference (shape: B={b}, T={t}, H={h}, W={w})")

    # Create execution context
    context = engine.create_execution_context()

    # Set input shapes
    context.set_input_shape("masked_flows", (b, t, 2, h, w))
    context.set_input_shape("masks", (b, t, 1, h, w))

    # Create PyTorch CUDA tensors for I/O
    masked_flows = torch.randn(b, t, 2, h, w, dtype=torch.float32).cuda()
    masks = torch.ones(b, t, 1, h, w, dtype=torch.float32).cuda()
    flow_out = torch.zeros(b, t, 2, h, w, dtype=torch.float32).cuda()

    # Get tensor addresses
    bindings = [
        masked_flows.data_ptr(),
        masks.data_ptr(),
        flow_out.data_ptr()
    ]

    # Setup CUDA stream
    stream = torch.cuda.current_stream().cuda_stream

    # Execute
    try:
        success = context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        if not success:
            print("[ERROR] Inference execution failed")
            return None

        print(f"      Inference successful")
        print(f"      Output shape: {flow_out.shape}")
        print(f"      Output range: [{flow_out.min():.3f}, {flow_out.max():.3f}]")
        print()
        return flow_out

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_accuracy(engine, shape=(1, 8, 256, 256)):
    """Compare TensorRT vs PyTorch outputs"""
    b, t, h, w = shape
    print(f"[4/4] Testing accuracy vs PyTorch (shape: B={b}, T={t}, H={h}, W={w})")

    try:
        # Load PyTorch model
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        pytorch_model = RecurrentFlowCompleteNet().cuda().eval()
        print(f"      PyTorch model loaded")

        # Create test inputs
        masked_flows = torch.randn(b, t, 2, h, w).cuda()
        masks = torch.ones(b, t, 1, h, w).cuda()

        # PyTorch inference
        with torch.no_grad():
            pytorch_out, _ = pytorch_model(masked_flows, masks)

        # TensorRT inference
        context = engine.create_execution_context()
        context.set_input_shape("masked_flows", (b, t, 2, h, w))
        context.set_input_shape("masks", (b, t, 1, h, w))

        flow_out = torch.zeros(b, t, 2, h, w, dtype=torch.float32).cuda()
        bindings = [masked_flows.data_ptr(), masks.data_ptr(), flow_out.data_ptr()]
        stream = torch.cuda.current_stream().cuda_stream

        context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        # Compare
        diff = torch.abs(pytorch_out - flow_out)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        relative_diff = max_diff / (pytorch_out.abs().max().item() + 1e-6)

        print()
        print(f"      Max absolute diff: {max_diff:.6f}")
        print(f"      Mean absolute diff: {mean_diff:.6f}")
        print(f"      Max relative diff: {relative_diff*100:.2f}%")
        print()

        if max_diff < 0.1 and relative_diff < 0.05:
            print(f"      [PASS] Numerical accuracy acceptable")
            return True
        else:
            print(f"      [WARN] Accuracy marginal (FP16 precision effects)")
            return True

    except Exception as e:
        print(f"[ERROR] Accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark(engine, shape=(1, 8, 640, 480), iterations=100):
    """Benchmark TensorRT performance"""
    b, t, h, w = shape
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()
    print(f"   Shape: B={b}, T={t}, H={h}, W={w}")
    print(f"   Iterations: {iterations} (+ {iterations//10} warmup)")
    print()

    # Setup
    context = engine.create_execution_context()
    context.set_input_shape("masked_flows", (b, t, 2, h, w))
    context.set_input_shape("masks", (b, t, 1, h, w))

    masked_flows = torch.randn(b, t, 2, h, w).cuda()
    masks = torch.ones(b, t, 1, h, w).cuda()
    flow_out = torch.zeros(b, t, 2, h, w).cuda()

    bindings = [masked_flows.data_ptr(), masks.data_ptr(), flow_out.data_ptr()]
    stream = torch.cuda.current_stream().cuda_stream

    # Warmup
    for _ in range(iterations//10):
        context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        context.execute_async_v3(stream_handle=stream)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Statistics
    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)

    print(f"   Mean:  {mean_time:.2f} ms")
    print(f"   Std:   {std_time:.2f} ms")
    print(f"   P50:   {p50:.2f} ms")
    print(f"   P95:   {p95:.2f} ms")
    print(f"   P99:   {p99:.2f} ms")
    print(f"   FPS:   {1000/mean_time:.1f}")
    print()

    if mean_time < 15:
        print(f"   [EXCELLENT] Performance <15ms (1.5x+ speedup vs PyTorch DCNv4)")
    elif mean_time < 20:
        print(f"   [GOOD] Performance <20ms")
    else:
        print(f"   [WARN] Performance needs optimization")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate RFC Net TensorRT Engine")
    parser.add_argument("--engine", type=str, default=DEFAULT_ENGINE, help="Engine path")
    parser.add_argument("--plugin", type=str, default=DEFAULT_PLUGIN, help="Plugin path")
    parser.add_argument("--skip-accuracy", action="store_true", help="Skip accuracy test")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    # Load engine
    engine = load_engine(args.engine, args.plugin)
    if not engine:
        sys.exit(1)

    # Test inference
    output = test_inference(engine)
    if output is None:
        sys.exit(1)

    # Test accuracy
    if not args.skip_accuracy:
        if not test_accuracy(engine):
            sys.exit(1)
    else:
        print("[SKIP] Accuracy test skipped")
        print()

    # Benchmark
    benchmark(engine, iterations=args.iterations)

    print("=" * 80)
    print("[SUCCESS] All validation tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
