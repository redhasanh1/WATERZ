"""
Benchmark FastFlowNet inference latency for both the original PyTorch
implementation and the TensorRT engine.

Usage:
    python tools/benchmark_fastflownet.py
        --shape 256                 # Spatial size (height = width)
        --iters 100                 # Timed iterations per backend
        --engine faster-propainter-main/engines/raft/raft_fp16.engine
        --trtexec D:/.../trtexec.exe  # Optional explicit trtexec path

Outputs average latency (milliseconds) for:
    - PyTorch FastFlowNet (FP32 on GPU)
    - TensorRT engine (via trtexec)

Prerequisites:
    - CUDA-capable GPU
    - PyTorch with CUDA
    - ptlflow installed
    - TensorRT CLI (trtexec) available in PATH or specified via --trtexec
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = REPO_ROOT / "faster-propainter-main" / "engines" / "raft" / "raft_fp16.engine"
DEFAULT_TRTEXEC_CANDIDATES = [
    # Explicit env override
    os.environ.get("TRTEXEC_PATH"),
    # Local TensorRT unpacked in repo
    (REPO_ROOT.parent / "TensorRT-10.7.0.23" / "bin" / "trtexec.exe").as_posix(),
    (REPO_ROOT.parent / "TensorRT-10.7.0.23" / "bin" / "trtexec").as_posix(),
    # Fallback to PATH lookup
    "trtexec",
]


def find_trtexec(explicit: Optional[str]) -> Optional[str]:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(DEFAULT_TRTEXEC_CANDIDATES)

    for cand in candidates:
        if not cand:
            continue
        # If it's an absolute/relative path, check existence; otherwise rely on shutil.which
        path_obj = Path(cand)
        if path_obj.is_file():
            return str(path_obj)
        if shutil_which(cand):
            return cand
    return None


def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


def benchmark_torch(shape: int, iters: int) -> tuple[float, str]:
    try:
        import ptlflow
    except ImportError as exc:
        raise RuntimeError(
            "ptlflow is required for the PyTorch benchmark. "
            "Install with `pip install ptlflow`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "[WARN] CUDA not available – running PyTorch benchmark on CPU. "
            "Timings will be much slower.",
            file=sys.stderr,
        )

    model = ptlflow.get_model("fastflownet", ckpt_path="things")
    model.to(device)
    model.eval()

    images = torch.randn(1, 2, 3, shape, shape, device=device, dtype=torch.float32)

    effective_iters = iters
    if device == "cpu" and iters > 50:
        effective_iters = 50  # keep runtime reasonable on CPU
        print(
            f"[INFO] Reducing PyTorch iterations to {effective_iters} for CPU timing.",
            file=sys.stderr,
        )

    warmup = max(10, iters // 10)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model({"images": images})["flows"]
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(effective_iters):
            _ = model({"images": images})["flows"]
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / effective_iters, device


def benchmark_trt(engine_path: Path, shape: int, iters: int, trtexec_cmd: str) -> float:
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    cmd = [
        trtexec_cmd,
        f"--loadEngine={engine_path.as_posix()}",
        f"--shapes=images:1x2x3x{shape}x{shape}",
        f"--iterations={iters}",
        "--warmUp=100",
        "--useSpinWait",
        "--avgRuns=10",
        "--memPoolSize=workspace:4096",
        "--verbose=0",
    ]

    env = os.environ.copy()
    trtexec_path = Path(trtexec_cmd)
    if trtexec_path.is_file():
        # Add sibling lib directory to PATH to resolve nvinfer_plugin_*.dll on Windows.
        bin_dir = trtexec_path.parent
        lib_dir = bin_dir.parent / "lib"
        path_entries = env.get("PATH", "").split(os.pathsep)
        if bin_dir.as_posix() not in path_entries:
            path_entries.insert(0, bin_dir.as_posix())
        if lib_dir.exists() and lib_dir.as_posix() not in path_entries:
            path_entries.insert(0, lib_dir.as_posix())
        env["PATH"] = os.pathsep.join(path_entries)

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        env=env,
    )

    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit code {completed.returncode}).\n{output}"
        )

    # Parse mean latency from summary line.
    match = re.search(r"mean = ([0-9.]+) ms", output)
    if not match:
        raise RuntimeError(
            "Could not parse latency from trtexec output. Full output:\n" + output
        )
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description="Benchmark FastFlowNet latency.")
    parser.add_argument(
        "--shape",
        type=int,
        default=256,
        help="Spatial resolution H=W used for benchmarking (default: 256).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Number of timed iterations per backend (default: 200).",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=DEFAULT_ENGINE,
        help=f"Path to TensorRT engine (default: {DEFAULT_ENGINE}).",
    )
    parser.add_argument(
        "--trtexec",
        type=str,
        default=None,
        help="Optional explicit path to trtexec binary.",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip PyTorch benchmark and only run TensorRT.",
    )
    parser.add_argument(
        "--skip-trt",
        action="store_true",
        help="Skip TensorRT benchmark and only run PyTorch.",
    )

    args = parser.parse_args()

    torch_latency = None
    trt_latency = None

    used_device = None
    if not args.skip_torch:
        try:
            torch_latency, used_device = benchmark_torch(args.shape, args.iters)
        except RuntimeError as exc:
            print(f"[WARN] Skipping PyTorch benchmark: {exc}", file=sys.stderr)
            torch_latency = None

    if not args.skip_trt:
        trtexec_cmd = find_trtexec(args.trtexec)
        if trtexec_cmd is None:
            raise RuntimeError(
                "Could not find `trtexec`. Specify --trtexec or add TensorRT "
                "bin directory to PATH."
            )
        trt_latency = benchmark_trt(args.engine, args.shape, args.iters, trtexec_cmd)

    print("==== FastFlowNet Latency (ms) ====")
    if torch_latency is not None:
        suffix = "CUDA" if used_device == "cuda" else "CPU"
        print(f"PyTorch (FastFlowNet, FP32, {suffix}): {torch_latency:.4f} ms")
    if trt_latency is not None:
        print(f"TensorRT (engine: {args.engine}): {trt_latency:.4f} ms")


if __name__ == "__main__":
    main()
