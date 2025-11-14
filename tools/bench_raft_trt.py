import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np


def _require(module: str):
    try:
        return __import__(module)
    except Exception as e:
        print(f"Missing dependency: {module}. Error: {e}")
        if module == "tensorrt":
            print("Install TensorRT Python package that matches your CUDA/TRT build.")
        if module.startswith("pycuda"):
            print("Install PyCUDA: pip install pycuda")
        sys.exit(1)


trt = _require("tensorrt")
cuda_drv = _require("pycuda.driver")
_ = _require("pycuda.autoinit")  # noqa: F401  (initializes CUDA context)


def load_engine(path: str) -> trt.ICudaEngine:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {path}")
    return engine


def _np_dtype(dt: trt.DataType):
    if dt == trt.DataType.HALF:
        return np.float16
    if dt == trt.DataType.FLOAT:
        return np.float32
    if dt == trt.DataType.INT32:
        return np.int32
    if dt == trt.DataType.INT8:
        return np.int8
    if dt == trt.DataType.BOOL:
        return np.bool_
    raise ValueError(f"Unsupported TRT dtype: {dt}")


def bench(engine_path: str, shapes: List[Tuple[int, int]], iters: int, warmup: int) -> None:
    engine = load_engine(engine_path)
    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("Failed to create execution context")

    # Resolve binding indices
    in_name = None
    out_name = None
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            in_name = name if in_name is None else in_name
        else:
            out_name = name if out_name is None else out_name
    if in_name is None or out_name is None:
        raise RuntimeError("Could not resolve input/output bindings")

    in_idx = engine.get_binding_index(in_name)
    out_idx = engine.get_binding_index(out_name)

    # Expect [B, T=2, C=3, H, W]
    batch = 1
    seq = 2
    chans = 3

    # Create stream
    stream = cuda_drv.Stream()

    print(f"Engine: {engine_path}")
    print(f"Input binding: {in_name} (index {in_idx}) | Output binding: {out_name} (index {out_idx})")

    for (h, w) in shapes:
        shape = (batch, seq, chans, h, w)
        # Set binding shape (profile 0)
        if not ctx.set_binding_shape(in_idx, shape):
            raise RuntimeError(f"Failed to set binding shape {shape}")

        # Query concrete shapes
        in_shape = tuple(ctx.get_binding_shape(in_idx))
        out_shape = tuple(ctx.get_binding_shape(out_idx))

        in_dtype = _np_dtype(engine.get_binding_dtype(in_idx))
        out_dtype = _np_dtype(engine.get_binding_dtype(out_idx))

        # Allocate host/device buffers
        inp = (np.random.rand(*in_shape).astype(in_dtype) * 2 - 1)
        d_in = cuda_drv.mem_alloc(inp.nbytes)
        out_nbytes = int(np.prod(out_shape) * np.dtype(out_dtype).itemsize)
        d_out = cuda_drv.mem_alloc(out_nbytes)

        bindings = [None] * engine.num_bindings
        bindings[in_idx] = int(d_in)
        bindings[out_idx] = int(d_out)

        # Warmup
        for _ in range(warmup):
            cuda_drv.memcpy_htod_async(d_in, inp, stream)
            ctx.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(iters):
            cuda_drv.memcpy_htod_async(d_in, inp, stream)
            ctx.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()
        dur = time.perf_counter() - start

        avg_ms = (dur / iters) * 1000.0
        fps = (iters / dur)
        print(f"H{h}xW{w} → avg {avg_ms:.3f} ms per pair | {fps:.2f} FPS (pairs/s)")


def parse_hw_list(values: List[str]) -> List[Tuple[int, int]]:
    shapes = []
    for v in values:
        v = v.lower().strip()
        if "x" in v:
            h, w = v.split("x", 1)
            shapes.append((int(h), int(w)))
        else:
            # Square
            d = int(v)
            shapes.append((d, d))
    return shapes


def main():
    p = argparse.ArgumentParser(description="Benchmark RAFT TensorRT engine (FastFlowNet)")
    p.add_argument("--engine", type=str, default=os.path.join(
        "faster-propainter-main", "engines", "raft", "raft_fp16.engine"
    ), help="Path to .engine file")
    p.add_argument("--shapes", nargs="*", default=["256", "640"], help="List like 256 or HxW")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    shapes = parse_hw_list(args.shapes)
    bench(args.engine, shapes, args.iters, args.warmup)


if __name__ == "__main__":
    main()

