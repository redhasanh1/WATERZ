"""
CUDA Graphs for Sparse Temporal Transformer
Eliminates kernel launch overhead for 1.2-1.5x speedup

Instead of fusing kernels (hard to beat cuBLAS/cuDNN), we batch all
transformer layers into a single CUDA graph to eliminate the ~10us
overhead per kernel launch.

For 8 layers × ~50 kernels/layer = 400 launches × 10us = 4ms overhead!
CUDA graphs reduce this to ~0ms (single graph launch).
"""
import torch
import torch.nn as nn
import os


class CUDAGraphWrapper(nn.Module):
    """
    Wraps a transformer model with CUDA graphs for faster inference

    Benefits:
    - Eliminates kernel launch overhead (~10us per kernel)
    - Reduces CPU overhead (graph replay is single API call)
    - Expected 1.2-1.5x speedup for transformer-heavy models

    Limitations:
    - Requires static shapes (no dynamic batch/sequence length)
    - Requires warmup to capture graph
    - Uses ~100MB additional VRAM per graph
    """
    def __init__(self, model, static_input_shape, warmup_iters=3, enable=True):
        super().__init__()
        self.model = model
        self.static_input_shape = static_input_shape
        self.warmup_iters = warmup_iters
        self.enable = enable and torch.cuda.is_available()

        # CUDA graph state
        self.graph = None
        self.static_input = None
        self.static_output = None
        self._captured = False
        self._logged = False

        if not self.enable:
            if not torch.cuda.is_available():
                print("[CUDA GRAPHS] CUDA not available - disabled")
            else:
                print("[CUDA GRAPHS] Disabled via enable=False")

    def _capture_graph(self, example_input, *args, **kwargs):
        """
        Capture CUDA graph for static execution

        This records all CUDA operations in the forward pass and
        replays them as a single graph (eliminates launch overhead).
        """
        if self._captured:
            return

        print(f"[CUDA GRAPHS] Capturing graph for shape {self.static_input_shape}...")

        # Allocate static tensors (reused for all graph replays)
        self.static_input = example_input.clone()

        # Warmup (required for graph capture)
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(self.warmup_iters):
                _ = self.model(self.static_input, *args, **kwargs)

        torch.cuda.synchronize()

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input, *args, **kwargs)

        self._captured = True
        print(f"[CUDA GRAPHS] Graph captured successfully!")

    def forward(self, x, *args, **kwargs):
        """
        Forward pass with optional CUDA graph acceleration

        If input shape matches static shape and graph is captured,
        replays the graph (fast!). Otherwise falls back to normal
        execution (first call or dynamic shapes).
        """
        # Fallback to normal execution if disabled
        if not self.enable:
            return self.model(x, *args, **kwargs)

        # Check if input shape matches static shape
        if x.shape != self.static_input_shape:
            if not self._logged:
                print(f"[CUDA GRAPHS] Dynamic shape {x.shape} != static {self.static_input_shape}, using fallback")
                self._logged = True
            return self.model(x, *args, **kwargs)

        # Capture graph on first forward pass
        if not self._captured:
            self._capture_graph(x, *args, **kwargs)

        # Copy input to static tensor (avoid allocation)
        self.static_input.copy_(x)

        # Replay graph (FAST! Single kernel launch)
        self.graph.replay()

        # Return static output (reused tensor)
        return self.static_output

    def reset(self):
        """Reset captured graph (e.g., after model weights change)"""
        self.graph = None
        self.static_input = None
        self.static_output = None
        self._captured = False
        print("[CUDA GRAPHS] Graph reset")


class TransformerCUDAGraphs:
    """
    Utility class for applying CUDA graphs to transformer models

    Usage:
        # Wrap transformer
        transformer = SparseTemporalTransformer(...)
        transformer_fast = TransformerCUDAGraphs.wrap(
            transformer,
            static_input_shape=(B, T, H, W, C)
        )

        # Use normally (first call captures graph)
        output = transformer_fast(x, fold_x_size, l_mask, t_dilation)
    """

    @staticmethod
    def wrap(model, static_input_shape, enable=None):
        """
        Wrap model with CUDA graphs

        Args:
            model: PyTorch model to wrap
            static_input_shape: Expected input shape (must be static!)
            enable: Force enable/disable (default: auto-detect from env)

        Returns:
            Wrapped model with CUDA graphs
        """
        if enable is None:
            enable = os.environ.get('ENABLE_CUDA_GRAPHS', '0') == '1'

        if not enable:
            print("[CUDA GRAPHS] Disabled via environment variable")
            return model

        return CUDAGraphWrapper(model, static_input_shape, enable=True)

    @staticmethod
    def benchmark(model, input_shape, num_iters=100, warmup=10):
        """
        Benchmark model with and without CUDA graphs

        Args:
            model: PyTorch model to benchmark
            input_shape: Input tensor shape
            num_iters: Number of iterations for timing
            warmup: Number of warmup iterations

        Returns:
            Dict with timing results
        """
        import time

        device = torch.device('cuda')
        model = model.to(device)
        x = torch.randn(*input_shape, device=device)

        print("\n" + "="*60)
        print("CUDA GRAPHS BENCHMARK")
        print("="*60)

        # Baseline (no CUDA graphs)
        print("\n1. Baseline (no CUDA graphs)")

        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = model(x)
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - t0) / num_iters * 1000

        print(f"  Time: {baseline_time:.3f} ms")

        # With CUDA graphs
        print("\n2. With CUDA graphs")

        model_graphed = CUDAGraphWrapper(model, input_shape, enable=True)

        for _ in range(warmup):
            _ = model_graphed(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(num_iters):
            _ = model_graphed(x)
        torch.cuda.synchronize()
        graphed_time = (time.perf_counter() - t0) / num_iters * 1000

        print(f"  Time: {graphed_time:.3f} ms")
        print(f"  Speedup: {baseline_time/graphed_time:.2f}x")

        print("\n" + "="*60)

        return {
            'baseline_ms': baseline_time,
            'graphed_ms': graphed_time,
            'speedup': baseline_time / graphed_time
        }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def apply_cuda_graphs_to_transformer(transformer, static_shape):
    """
    Apply CUDA graphs to SparseTemporalTransformer

    This wraps the transformer's forward pass to use CUDA graphs when
    the input shape is static (typical for production inference).

    Args:
        transformer: SparseTemporalTransformer instance
        static_shape: Expected input shape (B, T, H, W, C)

    Returns:
        Wrapped transformer with CUDA graphs enabled
    """
    enable = os.environ.get('ENABLE_CUDA_GRAPHS', '0') == '1'

    if not enable:
        print("[CUDA GRAPHS] Disabled - set ENABLE_CUDA_GRAPHS=1 to enable")
        return transformer

    # Create wrapper
    wrapped = CUDAGraphWrapper(
        transformer,
        static_input_shape=static_shape,
        warmup_iters=3,
        enable=True
    )

    print(f"[CUDA GRAPHS] Transformer wrapped for static shape {static_shape}")
    return wrapped


def benchmark_transformer_cuda_graphs():
    """
    Benchmark CUDA graphs on a simple transformer layer
    """
    from faster_propainter.model.modules.sparse_transformer import SparseWindowAttention

    # Create test transformer layer
    batch = 4
    T = 5  # temporal length
    H = 15  # after token merging: 15×15 → 7×7
    W = 15
    C = 512

    layer = SparseWindowAttention(
        dim=C,
        num_heads=4,
        window_size=(3, 8, 8),  # typical window size
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        use_checkpoint=False
    ).to('cuda', non_blocking=True)

    # Create dummy input
    input_shape = (batch, T, H, W, C)
    x = torch.randn(*input_shape, device='cuda')
    fold_x_size = (H, W)

    # Simple forward wrapper for benchmarking
    class TransformerWrapper(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            # Simplified forward (no mask, fixed fold_size)
            return self.layer(x, fold_x_size, l_mask=None, T_ind=0)

    model = TransformerWrapper(layer)

    # Benchmark
    results = TransformerCUDAGraphs.benchmark(
        model,
        input_shape=input_shape,
        num_iters=100,
        warmup=10
    )

    return results


if __name__ == "__main__":
    # Simple benchmark with a small model
    print("Testing CUDA Graphs...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Linear(512, 512),
            )

        def forward(self, x):
            return self.layers(x)

    model = SimpleModel()
    results = TransformerCUDAGraphs.benchmark(
        model,
        input_shape=(4, 225, 512),  # batch, seq_len, dim
        num_iters=100,
        warmup=10
    )

    print(f"\nResults: {results['baseline_ms']:.3f}ms -> {results['graphed_ms']:.3f}ms ({results['speedup']:.2f}x)")
