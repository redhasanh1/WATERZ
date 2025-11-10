"""
Test v3 FP16-constrained engine quality vs v2
"""

import os
import sys

trt_lib = r"D:\watermarkz\TensorRT-10.13.3.9\lib"
trt_bin = r"D:\watermarkz\TensorRT-10.13.3.9\bin"
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
os.environ['PATH'] = f"{trt_lib};{trt_bin};{cuda_bin};{os.environ['PATH']}"
os.add_dll_directory(trt_lib)
os.add_dll_directory(trt_bin)
os.add_dll_directory(cuda_bin)

import torch
import tensorrt as trt
import numpy as np

print("=" * 80)
print("Testing v3 FP16-Constrained Engine Quality")
print("=" * 80)
print()

# Test input
torch.manual_seed(42)
x = torch.randn(1, 11, 12, 12, 512, dtype=torch.float16, device='cuda')

print(f"Input: min={x.min().item():.4f} max={x.max().item():.4f} std={x.std().item():.4f}")
print()

def test_engine(engine_path, name):
    """Test engine and return output stats"""
    print(f"[{name}]")
    print(f"  Loading: {engine_path}")

    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())

    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()

    # Get I/O names
    in_name = None
    out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            in_name = name
        elif mode == trt.TensorIOMode.OUTPUT:
            out_name = name

    # Set input shape
    ctx.set_input_shape(in_name, (1, 11, 12, 12, 512))
    out_shape = tuple(ctx.get_tensor_shape(out_name))

    # CRITICAL: Check output dtype and allocate correct buffer
    output_dtype = engine.get_tensor_dtype(out_name)
    torch_dtype = torch.float32 if output_dtype == trt.DataType.FLOAT else torch.float16
    output = torch.empty(out_shape, dtype=torch_dtype, device='cuda')

    # Run inference
    with torch.cuda.stream(stream):
        ctx.set_tensor_address(in_name, int(x.data_ptr()))
        ctx.set_tensor_address(out_name, int(output.data_ptr()))
        ctx.execute_async_v3(stream.cuda_stream)

    stream.synchronize()

    # Stats
    out_min = output.min().item()
    out_max = output.max().item()
    out_mean = output.mean().item()
    out_std = output.std().item()

    print(f"  Output dtype: {torch_dtype}")
    print(f"  Output: min={out_min:.4f} max={out_max:.4f} mean={out_mean:.4f} std={out_std:.4f}")
    print()

    return out_std

# Test v2 (mixed precision with high std)
std_v2 = test_engine("engines/transformer/transformer_full_v2.engine", "v2 (Mixed Precision)")

# Test v3 (FP16-constrained)
std_v3 = test_engine("engines/transformer/transformer_full_v3_fp16_constrained.engine", "v3 (FP16-Constrained)")

# Compare
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"v2 (Mixed Precision):    std = {std_v2:.4f}")
print(f"v3 (FP16-Constrained):   std = {std_v3:.4f}")
print()

ratio = std_v2 / std_v3 if std_v3 > 0 else float('inf')
print(f"Improvement: {ratio:.2f}x reduction in std")
print()

if std_v3 < 2.5:
    print("✅ SUCCESS! v3 std < 2.5 (numerical stability achieved)")
else:
    print(f"❌ FAIL: v3 std = {std_v3:.4f} (expected < 2.5)")

print()
print("=" * 80)
