"""
Standalone Benchmark: RFCNet PyTorch vs TensorRT
Compares RecurrentFlowCompleteNet inference performance.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

import torch

# Ensure imports work
SCRIPT_DIR = Path(__file__).resolve().parent
FP_ROOT = SCRIPT_DIR / "faster-propainter-main"
if str(FP_ROOT) not in sys.path:
    sys.path.insert(0, str(FP_ROOT))


def load_pytorch_model(weights_path: str, device: torch.device):
    """Load PyTorch RFCNet model."""
    try:
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    except ModuleNotFoundError:
        import importlib.util as _ilu
        rfc_path = FP_ROOT / "model" / "recurrent_flow_completion.py"
        if not rfc_path.exists():
            raise
        spec = _ilu.spec_from_file_location("model.recurrent_flow_completion", rfc_path.as_posix())
        if spec is None or spec.loader is None:
            raise
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        RecurrentFlowCompleteNet = getattr(mod, "RecurrentFlowCompleteNet")

    model = RecurrentFlowCompleteNet(model_path=str(weights_path))
    model.to(device)
    model.eval()
    return model


def load_tensorrt_engine(engine_path: str):
    """Load TensorRT engine and create execution context."""
    try:
        import tensorrt as trt
        import ctypes
    except ImportError:
        print("[ERROR] TensorRT Python package not installed. Install with: pip install tensorrt")
        return None, None, None

    # CRITICAL: Load DCNv2 plugin DLL BEFORE deserializing engine
    plugin_path = Path(__file__).parent / "TensorRT-10.13.3.9" / "lib" / "mmdeploy_tensorrt_ops.dll"
    if not plugin_path.exists():
        print(f"[ERROR] DCNv2 plugin not found at: {plugin_path}")
        print(f"        Make sure mmdeploy_tensorrt_ops.dll exists in TensorRT-10.13.3.9/lib/")
        return None, None, None

    print(f"Loading DCNv2 plugin from: {plugin_path}")
    try:
        ctypes.CDLL(str(plugin_path))
        print("[OK] DCNv2 plugin loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load DCNv2 plugin: {e}")
        return None, None, None

    # Create logger
    logger = trt.Logger(trt.Logger.WARNING)

    # Initialize TensorRT plugin registry (required for custom plugins)
    trt.init_libnvinfer_plugins(logger, namespace="")

    # Load engine
    print(f"Loading TensorRT engine from: {engine_path}")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        print("[ERROR] Failed to deserialize TensorRT engine")
        return None, None, None

    context = engine.create_execution_context()
    if context is None:
        print("[ERROR] Failed to create execution context")
        return None, None, None

    print(f"[OK] TensorRT engine loaded successfully")
    print(f"     Engine inputs: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]}")
    print(f"     Engine outputs: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]}")

    return engine, context, runtime


def benchmark_pytorch(model, masked_flows, masks, device, warmup_iters=100, timed_iters=1000):
    """Benchmark PyTorch RFCNet inference."""
    print(f"\n[PyTorch Benchmark] Warmup: {warmup_iters} iterations, Timed: {timed_iters} iterations")

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(masked_flows, masks)

    # Timed iterations
    latencies = []
    with torch.no_grad():
        for _ in range(timed_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            flow, _ = model(masked_flows, masks)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }


def benchmark_tensorrt(engine, context, masked_flows, masks, warmup_iters=100, timed_iters=1000):
    """Benchmark TensorRT RFCNet inference."""
    import tensorrt as trt

    print(f"\n[TensorRT Benchmark] Warmup: {warmup_iters} iterations, Timed: {timed_iters} iterations")

    # Get tensor names
    input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]

    # Assuming standard naming from ONNX export
    masked_flows_name = "masked_flows"
    masks_name = "masks"
    flow_name = "flow"

    # Set input shapes
    context.set_input_shape(masked_flows_name, masked_flows.shape)
    context.set_input_shape(masks_name, masks.shape)

    # Allocate output buffer
    flow_shape = context.get_tensor_shape(flow_name)
    flow_output = torch.empty(tuple(flow_shape), dtype=torch.float32, device='cuda')

    # Set tensor addresses
    context.set_tensor_address(masked_flows_name, masked_flows.data_ptr())
    context.set_tensor_address(masks_name, masks.data_ptr())
    context.set_tensor_address(flow_name, flow_output.data_ptr())

    # Warmup
    for _ in range(warmup_iters):
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # Timed iterations
    latencies = []
    for _ in range(timed_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    return {
        'mean': np.mean(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p50': np.percentile(latencies, 50),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
    }


def print_results(pytorch_stats, tensorrt_stats):
    """Print benchmark results and speedup comparison."""
    print("\n" + "="*80)
    print(" BENCHMARK RESULTS: RFCNet PyTorch vs TensorRT")
    print("="*80)

    print("\nPyTorch Performance:")
    print(f"  Mean Latency:   {pytorch_stats['mean']:.3f} ms")
    print(f"  Std Dev:        {pytorch_stats['std']:.3f} ms")
    print(f"  Min Latency:    {pytorch_stats['min']:.3f} ms")
    print(f"  Max Latency:    {pytorch_stats['max']:.3f} ms")
    print(f"  P50 (Median):   {pytorch_stats['p50']:.3f} ms")
    print(f"  P95:            {pytorch_stats['p95']:.3f} ms")
    print(f"  P99:            {pytorch_stats['p99']:.3f} ms")
    print(f"  Throughput:     {1000.0 / pytorch_stats['mean']:.2f} fps")

    print("\nTensorRT Performance:")
    print(f"  Mean Latency:   {tensorrt_stats['mean']:.3f} ms")
    print(f"  Std Dev:        {tensorrt_stats['std']:.3f} ms")
    print(f"  Min Latency:    {tensorrt_stats['min']:.3f} ms")
    print(f"  Max Latency:    {tensorrt_stats['max']:.3f} ms")
    print(f"  P50 (Median):   {tensorrt_stats['p50']:.3f} ms")
    print(f"  P95:            {tensorrt_stats['p95']:.3f} ms")
    print(f"  P99:            {tensorrt_stats['p99']:.3f} ms")
    print(f"  Throughput:     {1000.0 / tensorrt_stats['mean']:.2f} fps")

    speedup = pytorch_stats['mean'] / tensorrt_stats['mean']
    print("\nSpeedup Analysis:")
    print(f"  TensorRT is {speedup:.2f}x faster than PyTorch (mean latency)")
    print(f"  Absolute speedup: {pytorch_stats['mean'] - tensorrt_stats['mean']:.3f} ms saved per inference")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RFCNet PyTorch vs TensorRT")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/recurrent_flow_completion.pth",
        help="Path to PyTorch weights"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="faster-propainter-main/engines/rfcnet/rfcnet_fp16.engine",
        help="Path to TensorRT engine"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="1x8x2x480x480",
        help="Input shape as BxTxCxHxW (e.g., 1x8x2x480x480)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of timed iterations"
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Benchmark PyTorch only (skip TensorRT)"
    )
    parser.add_argument(
        "--tensorrt-only",
        action="store_true",
        help="Benchmark TensorRT only (skip PyTorch)"
    )

    args = parser.parse_args()

    # Parse input shape
    try:
        b, t, c, h, w = map(int, args.shape.lower().replace(" ", "").split("x"))
        print(f"\nInput Shape: B={b}, T={t}, C={c}, H={h}, W={w}")
    except ValueError:
        print(f"[ERROR] Invalid shape format: {args.shape}. Expected BxTxCxHxW (e.g., 1x8x2x480x480)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create dummy inputs
    masked_flows = torch.zeros((b, t, 2, h, w), dtype=torch.float32, device=device)
    masks = torch.zeros((b, t, 1, h, w), dtype=torch.float32, device=device)
    print(f"Input tensors created:")
    print(f"  masked_flows: {masked_flows.shape}")
    print(f"  masks: {masks.shape}")

    pytorch_stats = None
    tensorrt_stats = None

    # Benchmark PyTorch
    if not args.tensorrt_only:
        print("\n" + "-"*80)
        print(" PYTORCH BENCHMARK")
        print("-"*80)
        try:
            model = load_pytorch_model(args.weights, device)
            print(f"[OK] PyTorch model loaded from: {args.weights}")
            pytorch_stats = benchmark_pytorch(
                model, masked_flows, masks, device,
                warmup_iters=args.warmup,
                timed_iters=args.iterations
            )
            print(f"[DONE] PyTorch benchmark complete: {pytorch_stats['mean']:.3f} ms mean latency")
        except Exception as e:
            print(f"[ERROR] PyTorch benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # Benchmark TensorRT
    if not args.pytorch_only:
        print("\n" + "-"*80)
        print(" TENSORRT BENCHMARK")
        print("-"*80)
        try:
            engine, context, runtime = load_tensorrt_engine(args.engine)
            if engine is not None:
                tensorrt_stats = benchmark_tensorrt(
                    engine, context, masked_flows, masks,
                    warmup_iters=args.warmup,
                    timed_iters=args.iterations
                )
                print(f"[DONE] TensorRT benchmark complete: {tensorrt_stats['mean']:.3f} ms mean latency")
            else:
                print("[ERROR] Failed to load TensorRT engine, skipping TensorRT benchmark")
        except Exception as e:
            print(f"[ERROR] TensorRT benchmark failed: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if pytorch_stats is not None and tensorrt_stats is not None:
        print_results(pytorch_stats, tensorrt_stats)
    elif pytorch_stats is not None:
        print(f"\nPyTorch-only result: {pytorch_stats['mean']:.3f} ms mean latency")
    elif tensorrt_stats is not None:
        print(f"\nTensorRT-only result: {tensorrt_stats['mean']:.3f} ms mean latency")
    else:
        print("\n[ERROR] No benchmarks completed successfully")


if __name__ == "__main__":
    main()
