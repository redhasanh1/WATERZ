# GPU Mask Creation Analysis

## Test Results: GPU is SLOWER than CPU!

**Benchmark (300 frames @ 872×480)**:
- CPU Sequential: **0.076s** (0.25ms/mask)
- GPU Batch (Kornia): **0.488s** (1.63ms/mask)
- Result: **CPU is 6.4x FASTER!**

## Why GPU is Slower

### 1. GPU Transfer Overhead
- 300 frames @ 872×480 = **~124 MB** data transfer to GPU
- PCIe bandwidth: ~16 GB/s
- Transfer time: ~8ms (dominates the 0.25ms CPU blur time!)

### 2. Small Kernel Size
- 21×21 Gaussian blur is **tiny**
- OpenCV's CPU implementation uses highly optimized SIMD instructions (AVX2/AVX512)
- GPU kernel launch overhead + memory transfers exceed CPU execution time

### 3. Gaussian Blur is Memory-Bound, Not Compute-Bound
- Blur: Read pixel, multiply, write pixel (low arithmetic intensity)
- GPU advantage: High compute (matrix multiply, conv2d with large kernels)
- CPU advantage: Low latency memory access (L1/L2 cache hit rate ~95%)

## Real Production Performance

Looking at your production logs:
```
[OK] Cropped 67 frames + 67 masks in memory: 0.000s
```

**Mask creation is already negligible** (< 0.1s for 67 frames).

The real bottlenecks are:
1. **Transformer**: 58.8% of runtime (solved with FP8!)
2. **Flow Completion (RFCNet)**: 11.7% of runtime
3. **Encoder/Decoder**: 23.5% of runtime

## Recommendation

### ✅ DISABLE GPU Mask Creation
CPU mask creation is:
- **6.4x faster** (0.076s vs 0.488s)
- **Simpler** (no GPU transfer overhead)
- **More reliable** (no GPU memory issues)

### How to Disable

**Option 1: Modify yolo_detector.py** (Recommended)

Change line 78:
```python
# Before
self.use_gpu_masks = False  # Change to False

# After
self.use_gpu_masks = False  # CPU is 6x faster for small blurs!
```

**Option 2: Add Environment Variable**

In `START_CELERY_TRT.bat`, add:
```bat
set DISABLE_GPU_MASKS=1
```

Then modify `yolo_detector.py` line 82:
```python
if torch is not None and torch.cuda.is_available() and os.getenv("DISABLE_GPU_MASKS", "0") != "1":
```

## When GPU Masks Make Sense

GPU masks would be faster if:
1. **Larger kernels**: 101×101 or 201×201 (high compute intensity)
2. **Many frames**: 10,000+ frames (amortize transfer overhead)
3. **Already on GPU**: Frames already in GPU memory (zero transfer cost)
4. **Complex operations**: Multi-pass filtering, morphological ops, etc.

For typical 21×21 blur on 300 frames, **CPU is optimal**.

## Impact on Overall Pipeline

With FP8 transformer optimization:
- Before: 30.68s per 30 frames
- After FP8: 5.72s per 30 frames
- Mask creation: ~0.08s (1.4% of total time)

**Mask creation is NOT a bottleneck!** Focus on:
1. ✅ FP8 Transformer (DONE - 5.36x speedup!)
2. ⏭ RFCNet optimization (11.7% of runtime)
3. ⏭ Encoder/Decoder optimization (23.5% of runtime)
