"""
RAFT Benchmark: PyTorch RAFT_bi vs TensorRT FastFlowNet engine

Usage:
  python benchmark_raft.py <path_to_video> [--iters 20] [--max_frames 120] [--height 480]

Notes:
- Looks for TensorRT engine at faster-propainter-main/engines/raft/raft_fp16.engine
- Falls back to PyTorch RAFT if engine not found or TensorRT fails to load
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch


def add_repo_paths():
    # Ensure faster-propainter-main is importable
    root = os.getcwd()
    cand = os.path.join(root, "faster-propainter-main")
    if cand not in sys.path:
        sys.path.append(cand)


def read_video_frames(path: str, max_frames: int, target_h: int = 480) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert BGR to RGB for consistency with pipeline conventions
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read from input video")

    # Resize to target height (keep aspect), ensure dims divisible by 8
    h, w = frames[0].shape[:2]
    if target_h > 0 and h != target_h:
        scale = target_h / float(h)
        new_w = int(round(w * scale))
        frames = [cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA) for f in frames]

    # Make dimensions divisible by 8
    h, w = frames[0].shape[:2]
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    if (h8, w8) != (h, w):
        frames = [cv2.resize(f, (w8, h8), interpolation=cv2.INTER_AREA) for f in frames]
    return frames


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    # [T, H, W, C] uint8 -> [1, T, C, H, W] float in [-1, 1]
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)  # TCHW
    ten = torch.from_numpy(arr)[None, ...]  # 1TCHW
    ten = ten * 2.0 - 1.0
    return ten


class TRTOpticalFlow:
    def __init__(self, device: torch.device, engine_path: str):
        self.device = device
        self._engine = None
        self._ctx = None
        self._in_idx = None
        self._out_idx = None
        self._in_dtype = None
        self._out_dtype = None

        # Try to make TRT DLLs available on Windows
        trt_root = os.environ.get("TENSORRT_ROOT") or os.path.join(os.getcwd(), "TensorRT-10.13.3.9")
        trt_lib = os.path.join(trt_root, "lib")
        if os.name == "nt" and os.path.isdir(trt_lib):
            try:
                os.add_dll_directory(trt_lib)  # type: ignore[attr-defined]
            except Exception:
                pass

        import tensorrt as trt  # type: ignore

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        self._engine = engine
        self._ctx = engine.create_execution_context()
        # Support legacy bindings API and TensorRT 10 v3 I/O API
        if hasattr(engine, "num_bindings"):
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i) and self._in_idx is None:
                    self._in_idx = i
                    self._in_dtype = engine.get_binding_dtype(i)
                if (not engine.binding_is_input(i)) and self._out_idx is None:
                    self._out_idx = i
                    self._out_dtype = engine.get_binding_dtype(i)
            if self._in_idx is None or self._out_idx is None:
                raise RuntimeError("Could not resolve input/output bindings (legacy)")
        else:
            import tensorrt as trt  # type: ignore
            self._in_name = None
            self._out_name = None
            n_io = engine.num_io_tensors
            for i in range(n_io):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT and self._in_name is None:
                    self._in_name = name
                    self._in_dtype = engine.get_tensor_dtype(name)
                elif mode == trt.TensorIOMode.OUTPUT and self._out_name is None:
                    self._out_name = name
                    self._out_dtype = engine.get_tensor_dtype(name)
            if self._in_name is None or self._out_name is None:
                raise RuntimeError("Could not resolve input/output tensor names (v3)")

    @torch.no_grad()
    def _exec_pair(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        # img0/img1: [B,C,H,W] on device
        import tensorrt as trt  # type: ignore
        in_half = (self._in_dtype == trt.DataType.HALF)
        out_half = (self._out_dtype == trt.DataType.HALF)

        inp = torch.stack((img0, img1), dim=1)  # [B,2,3,H,W]
        inp = inp.half() if in_half else inp.float()
        b, _, _, h, w = inp.shape
        out = torch.empty((b, 2, h, w), device=inp.device, dtype=torch.float16 if out_half else torch.float32)
        if hasattr(self._engine, "num_bindings"):
            self._ctx.set_binding_shape(self._in_idx, tuple(inp.shape))
            bindings = [None] * self._engine.num_bindings
            bindings[self._in_idx] = int(inp.data_ptr())
            bindings[self._out_idx] = int(out.data_ptr())
            ok = self._ctx.execute_v2(bindings)
        else:
            self._ctx.set_input_shape(self._in_name, tuple(inp.shape))
            self._ctx.set_tensor_address(self._in_name, int(inp.data_ptr()))
            self._ctx.set_tensor_address(self._out_name, int(out.data_ptr()))
            stream_ptr = torch.cuda.current_stream().cuda_stream
            ok = self._ctx.execute_async_v3(stream_ptr)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed")
        return out

    @torch.no_grad()
    def run_sequence(self, frames_btchw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # frames: [1,T,C,H,W]
        assert frames_btchw.dim() == 5 and frames_btchw.size(0) == 1
        _, T, _, _, _ = frames_btchw.shape
        dev_frames = frames_btchw.to(self.device)
        flows_f = []
        flows_b = []
        for t in range(T - 1):
            f0 = dev_frames[:, t]
            f1 = dev_frames[:, t + 1]
            ff = self._exec_pair(f0, f1)
            fb = self._exec_pair(f1, f0)
            flows_f.append(ff)
            flows_b.append(fb)
        flows_f = torch.stack(flows_f, dim=0).permute(1, 0, 2, 3, 4)  # [1,T-1,2,H,W]
        flows_b = torch.stack(flows_b, dim=0).permute(1, 0, 2, 3, 4)
        return flows_f, flows_b


def benchmark_pytorch_raft(frames_btchw: torch.Tensor, device: torch.device, iters: int) -> Tuple[float, float]:
    add_repo_paths()
    from model.modules.flow_comp_raft import RAFT_bi  # type: ignore

    ckpt = os.path.join("weights", "raft-things.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"RAFT weights not found: {ckpt}")

    model = RAFT_bi(ckpt, device if device.type == "cuda" else "cpu")
    model.eval()
    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = False
    except Exception:
        pass
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # Some RAFT/cuDNN combinations error on certain tensor layouts.
        # Disable cuDNN during the call to avoid CUDNN_STATUS_NOT_SUPPORTED.
        try:
            import torch.backends.cudnn as cudnn
            prev_enabled = cudnn.enabled
            cudnn.enabled = False
        except Exception:
            prev_enabled = None
        try:
            flows_f, flows_b = model(frames_btchw.to(device).contiguous(), iters=iters)
            _ = (flows_f.mean() + flows_b.mean()).item()
        finally:
            if prev_enabled is not None:
                cudnn.enabled = prev_enabled
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    # frames processed = number of pairs (T-1)
    T = frames_btchw.size(1)
    fps = (T - 1) / dt if dt > 0 else 0.0
    return dt, fps


def benchmark_trt_fastflownet(frames_btchw: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    engine_candidates = [
        os.path.join("faster-propainter-main", "engines", "raft", "raft_fp16.engine"),
    ]
    engine_path = None
    for p in engine_candidates:
        if os.path.exists(p):
            engine_path = p
            break
    if engine_path is None:
        raise FileNotFoundError("TensorRT RAFT engine not found at faster-propainter-main/engines/raft/raft_fp16.engine")

    trt_runner = TRTOpticalFlow(device, engine_path)

    # Warmup a few pairs
    T = frames_btchw.size(1)
    if T >= 3:
        _ = trt_runner.run_sequence(frames_btchw[:, :3])
        if device.type == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    flows_f, flows_b = trt_runner.run_sequence(frames_btchw)
    # Touch
    _ = (flows_f.mean() + flows_b.mean()).item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fps = (T - 1) / dt if dt > 0 else 0.0
    return dt, fps


def main():
    parser = argparse.ArgumentParser(description="Benchmark RAFT: PyTorch vs TensorRT")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--iters", type=int, default=20, help="RAFT iterations")
    parser.add_argument("--max_frames", type=int, default=120, help="Max frames to process from video")
    parser.add_argument("--height", type=int, default=480, help="Resize video height before processing")
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"[!] Video not found: {video_path}")
        sys.exit(1)

    print("".ljust(10) + "RAfT BENCHMARK: PyTorch vs TensorRT")
    print(f"Input: {video_path}")

    frames = read_video_frames(video_path, max_frames=args.max_frames, target_h=args.height)
    frames_btchw = frames_to_tensor(frames)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frames: {frames_btchw.size(1)}  Size: {frames_btchw.size(-1)}x{frames_btchw.size(-2)}  Iters: {args.iters}")

    results = {}

    # Test 1: PyTorch RAFT_bi
    try:
        print("\nTEST 1: PyTorch RAFT_bi")
        dt, fps = benchmark_pytorch_raft(frames_btchw, device, args.iters)
        results["pytorch_raft"] = {"duration": dt, "fps": fps}
        print(f"  Duration: {dt:.2f}s   Pairs/s: {fps:.1f}")
    except Exception as e:
        results["pytorch_raft"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # Test 2: TensorRT FastFlowNet engine
    try:
        print("\nTEST 2: TensorRT FastFlowNet (.engine)")
        dt, fps = benchmark_trt_fastflownet(frames_btchw, device)
        results["tensorrt_fastflownet"] = {"duration": dt, "fps": fps}
        print(f"  Duration: {dt:.2f}s   Pairs/s: {fps:.1f}")
    except Exception as e:
        results["tensorrt_fastflownet"] = {"error": str(e)}
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + " ".ljust(10) + "BENCHMARK RESULTS")
    print(f"{'Variant':<28} {'Time (s)':>10} {'Pairs/s':>12}")
    if "pytorch_raft" in results and "duration" in results["pytorch_raft"]:
        r = results["pytorch_raft"]
        print(f"{'PyTorch RAFT_bi':<28} {r['duration']:>10.2f} {r['fps']:>12.1f}")
    else:
        print(f"{'PyTorch RAFT_bi':<28} {'N/A':>10} {'N/A':>12}")
    if "tensorrt_fastflownet" in results and "duration" in results["tensorrt_fastflownet"]:
        r = results["tensorrt_fastflownet"]
        print(f"{'TensorRT FastFlowNet':<28} {r['duration']:>10.2f} {r['fps']:>12.1f}")
    else:
        print(f"{'TensorRT FastFlowNet':<28} {'N/A':>10} {'N/A':>12}")


if __name__ == "__main__":
    main()
