import argparse
import time
from typing import List, Tuple

import torch


def parse_hw_list(values: List[str]) -> List[Tuple[int, int]]:
    shapes = []
    for v in values:
        v = v.lower().strip()
        if "x" in v:
            h, w = v.split("x", 1)
            shapes.append((int(h), int(w)))
        else:
            d = int(v)
            shapes.append((d, d))
    return shapes


def build_model(device: torch.device):
    import ptlflow

    model = ptlflow.get_model("fastflownet", ckpt_path="things")
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def bench(iters: int, warmup: int, shapes: List[Tuple[int, int]], device: torch.device, use_fp16: bool):
    model = build_model(device)

    dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32

    for (h, w) in shapes:
        # [B, T=2, C=3, H, W]
        dummy = torch.rand(1, 2, 3, h, w, device=device, dtype=dtype) * 2 - 1

        # Warmup
        for _ in range(warmup):
            _ = model({"images": dummy})
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timed
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model({"images": dummy})
            if device.type == "cuda":
                torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        avg_ms = (dt / iters) * 1000.0
        fps = iters / dt
        print(f"Torch FastFlowNet H{h}xW{w} avg {avg_ms:.3f} ms per pair | {fps:.2f} FPS")


def main():
    p = argparse.ArgumentParser(description="Benchmark FastFlowNet in PyTorch")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--shapes", nargs="*", default=["256", "640"], help="List like 256 or HxW")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes = parse_hw_list(args.shapes)
    bench(args.iters, args.warmup, shapes, device, args.fp16)


if __name__ == "__main__":
    main()
