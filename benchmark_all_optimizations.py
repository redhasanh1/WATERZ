"""
Comprehensive Benchmark: ALL Transformer Optimizations
Tests cumulative speedup from each optimization layer

Optimizations tested:
1. Baseline FP32 (no optimizations)
2. + Flash Attention (PyTorch SDPA)
3. + FP8 quantization (RTX 4090 Ada)
4. + torch.compile (kernel fusion)

Target: 551ms → 50-100ms (5-10x speedup)
"""
import os
import sys
import torch
import time
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, 'faster-propainter-main')
from model.modules.sparse_transformer import TemporalSparseTransformerBlock

def create_transformer():
    """Create transformer with t2t_params"""
    t2t_params = {
        'kernel_size': (7, 7),
        'stride': (3, 3),
        'padding': (3, 3)
    }

    model = TemporalSparseTransformerBlock(
        dim=512,
        n_head=4,
        window_size=(5, 9),
        pool_size=(4, 4),
        depths=8,
        t2t_params=t2t_params
    )
    return model

def benchmark_config(config_name, env_vars, num_iterations=100, warmup=10):
    """
    Benchmark a specific configuration

    Args:
        config_name: Name of configuration
        env_vars: Dict of environment variables to set
        num_iterations: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        dict: Benchmark results
    """
    print("\n" + "=" * 80)
    print(f"BENCHMARKING: {config_name}")
    print("=" * 80)

    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = str(value)
        print(f"[ENV] {key}={value}")

    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    device = torch.device("cuda")

    # Create model (will read env vars)
    print(f"\n[MODEL] Creating transformer...")
    model = create_transformer()
    model = model.to(device)
    model.eval()

    # Create input
    batch_size = 1
    T, H, W, C = 60, 20, 36, 512

    print(f"[INPUT] Shape: [{batch_size}, {T}, {H}, {W}, {C}]")
    x = torch.randn(batch_size, T, H, W, C, device=device, dtype=torch.float32)
    fold_size = (60, 108)
    mask = torch.ones(batch_size, T, H, W, 1, device=device, dtype=torch.float32)

    # Warmup
    print(f"\n[WARMUP] Running {warmup} iterations...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for i in range(warmup):
            _ = model(x, fold_size, mask)
    torch.cuda.synchronize()
    print("[OK] Warmup complete")

    # Benchmark
    print(f"\n[BENCHMARK] Running {num_iterations} iterations...")
    timings = []

    torch.cuda.synchronize()
    with torch.no_grad():
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x, fold_size, mask)

            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append((end - start) * 1000)

            if (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{num_iterations} complete")

    # Statistics
    timings = np.array(timings)
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    p50 = np.percentile(timings, 50)
    p95 = np.percentile(timings, 95)

    # GPU Memory
    max_memory = torch.cuda.max_memory_allocated(device) / 1024**3

    # Results
    print(f"\n[RESULTS] {config_name}:")
    print(f"  Mean:   {mean_time:.3f} ms ± {std_time:.3f} ms")
    print(f"  Median: {p50:.3f} ms")
    print(f"  Min:    {min_time:.3f} ms")
    print(f"  P95:    {p95:.3f} ms")
    print(f"  Memory: {max_memory:.3f} GB")

    # Clean up
    del model, x, mask
    torch.cuda.empty_cache()

    return {
        'config': config_name,
        'mean_ms': float(mean_time),
        'std_ms': float(std_time),
        'min_ms': float(min_time),
        'p50_ms': float(p50),
        'p95_ms': float(p95),
        'gpu_memory_gb': float(max_memory)
    }

def main():
    """Run comprehensive benchmark suite"""

    print("=" * 80)
    print("COMPREHENSIVE TRANSFORMER OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 80)

    results = []

    # Configuration 1: Pure FP32 baseline (NO optimizations)
    print("\n\nCONFIGURATION 1/4: Baseline FP32")
    print("Optimizations: NONE")
    result1 = benchmark_config(
        "1. Baseline FP32",
        {
            'ENABLE_FLASH_ATTENTION': '0',
            'ENABLE_FP8_TRANSFORMER': '0',
            'USE_TORCH_COMPILE': '0'
        },
        num_iterations=100,
        warmup=10
    )
    results.append(result1)

    # Configuration 2: FP32 + Flash Attention
    print("\n\nCONFIGURATION 2/4: + Flash Attention")
    print("Optimizations: Flash Attention (PyTorch SDPA)")
    result2 = benchmark_config(
        "2. + Flash Attention",
        {
            'ENABLE_FLASH_ATTENTION': '1',
            'ENABLE_FP8_TRANSFORMER': '0',
            'USE_TORCH_COMPILE': '0'
        },
        num_iterations=100,
        warmup=10
    )
    results.append(result2)

    # Configuration 3: FP32 + Flash Attention + FP8
    print("\n\nCONFIGURATION 3/4: + FP8 Quantization")
    print("Optimizations: Flash Attention + FP8 (RTX 4090 Ada)")
    result3 = benchmark_config(
        "3. + FP8 Quantization",
        {
            'ENABLE_FLASH_ATTENTION': '1',
            'ENABLE_FP8_TRANSFORMER': '1',
            'USE_TORCH_COMPILE': '0'
        },
        num_iterations=100,
        warmup=10
    )
    results.append(result3)

    # Configuration 4: ALL optimizations (Flash + FP8 + torch.compile)
    print("\n\nCONFIGURATION 4/4: + torch.compile")
    print("Optimizations: Flash Attention + FP8 + torch.compile (max-autotune)")
    result4 = benchmark_config(
        "4. + torch.compile",
        {
            'ENABLE_FLASH_ATTENTION': '1',
            'ENABLE_FP8_TRANSFORMER': '1',
            'USE_TORCH_COMPILE': '1'
        },
        num_iterations=100,
        warmup=10
    )
    results.append(result4)

    # Summary
    print("\n\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print()

    baseline_time = results[0]['mean_ms']

    print(f"{'Configuration':<30} {'Mean (ms)':<12} {'Speedup':<10} {'vs Baseline':<12}")
    print("-" * 80)

    for i, result in enumerate(results):
        mean_ms = result['mean_ms']
        speedup_vs_prev = results[i-1]['mean_ms'] / mean_ms if i > 0 else 1.0
        speedup_vs_baseline = baseline_time / mean_ms

        print(f"{result['config']:<30} {mean_ms:>8.2f} ms   {speedup_vs_prev:>5.2f}x      {speedup_vs_baseline:>6.2f}x")

    print("\n" + "=" * 80)
    print("CUMULATIVE SPEEDUP")
    print("=" * 80)

    final_speedup = baseline_time / results[-1]['mean_ms']
    time_saved = baseline_time - results[-1]['mean_ms']
    percent_saved = (time_saved / baseline_time) * 100

    print(f"\nBaseline (FP32):                     {baseline_time:.2f} ms")
    print(f"Fully Optimized:                     {results[-1]['mean_ms']:.2f} ms")
    print(f"\n{'TOTAL SPEEDUP:':<30} {final_speedup:.2f}x")
    print(f"{'Time Saved:':<30} {time_saved:.2f} ms ({percent_saved:.1f}%)")

    # Save results
    results_path = Path('optimization_benchmark_results.txt')
    with open(results_path, 'w') as f:
        f.write("COMPREHENSIVE TRANSFORMER OPTIMIZATION BENCHMARK\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Hardware: {torch.cuda.get_device_name(0)}\n")
        f.write(f"CUDA: {torch.version.cuda}\n")
        f.write(f"PyTorch: {torch.__version__}\n\n")

        f.write("RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Configuration':<30} {'Mean (ms)':<12} {'Speedup':<10} {'vs Baseline':<12}\n")
        f.write("-" * 80 + "\n")

        for i, result in enumerate(results):
            mean_ms = result['mean_ms']
            speedup_vs_prev = results[i-1]['mean_ms'] / mean_ms if i > 0 else 1.0
            speedup_vs_baseline = baseline_time / mean_ms

            f.write(f"{result['config']:<30} {mean_ms:>8.2f} ms   {speedup_vs_prev:>5.2f}x      {speedup_vs_baseline:>6.2f}x\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CUMULATIVE RESULTS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nBaseline (FP32): {baseline_time:.2f} ms\n")
        f.write(f"Fully Optimized: {results[-1]['mean_ms']:.2f} ms\n")
        f.write(f"\nTotal Speedup: {final_speedup:.2f}x\n")
        f.write(f"Time Saved: {time_saved:.2f} ms ({percent_saved:.1f}%)\n")

    print(f"\n[SAVED] Results saved to: {results_path}")
    print("\n[BENCHMARK COMPLETE]")

if __name__ == "__main__":
    main()
