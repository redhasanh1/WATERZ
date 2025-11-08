# GPU Optimization Analysis - Findings & Strategy

## Executive Summary

**Key Discovery**: Mask creation is **NOT** a bottleneck (only 0.38s for 300 frames). GPU acceleration makes it **6.6x SLOWER** due to transfer overhead.

## Benchmark Results

### Mask Creation Performance

| Method | Time | ms/frame | Speedup |
|--------|------|----------|---------|
| **CPU (cv2.GaussianBlur)** | 0.384s | 1.28ms | Baseline |
| **GPU (Kornia + transfer)** | 2.555s | 8.52ms | **0.15x (6.6x SLOWER!)** |

**Why GPU Failed:**
- Transfer overhead: 600MB CPU→GPU + 600MB GPU→CPU for 300 x 1080p masks
- CPU cv2.GaussianBlur is already highly optimized (SIMD, multithreading)
- 21x21 Gaussian blur is not complex enough to justify GPU transfer

### INT8 TensorRT Performance

| Engine | Size | Status |
|--------|------|--------|
| **FP16** | 9.8 MB | ✅ Working at **946 fps** (1.06ms/frame) |
| **INT8** | 7.97 MB | ❌ CUDA errors with non-square inputs |

**INT8 Issues:**
- Built with trtexec expecting 640x640 square inputs
- Crashes with "illegal memory access" on letterboxed inputs (640x352)
- Dynamic range quantization without calibration may be unstable
- **Not worth fixing** - FP16 is already excellent

## Revised Optimization Strategy

### ❌ DON'T Optimize (Already Fast)

1. **Mask Creation**: 0.38s is already fast, GPU makes it worse
2. **INT8 YOLO**: FP16 at 946 fps is excellent, INT8 has bugs
3. **GPU Preprocessing**: Only 0.1s, transfer overhead >> computation

### ✅ DO Optimize (Real Gains)

1. **GPU Video Decode (NVDEC)** - Priority 1
   - Current: 0.17s (CPU H.264 decode)
   - Target: 0.03-0.05s (GPU decode)
   - **Estimated gain**: 0.12-0.14s per 300 frames
   - **Implementation**: nvdec_video_loader.py (created ✅)

2. **Future: Zero-Copy GPU Pipeline**
   - Decode frames on GPU → keep as CUDA tensors
   - YOLO inference on GPU (already done)
   - GPU masks on GPU (zero-copy, no transfer)
   - ProPainter on GPU (already done)
   - **Eliminates all CPU↔GPU transfers!**

## Current Pipeline Timing (300 frames)

```
Frame Loading (CPU decode):     0.17s  ← OPTIMIZE THIS (NVDEC)
YOLO Detection (FP16 TensorRT):  0.32s  ✅ ALREADY OPTIMAL (946 fps)
Mask Creation (CPU):             0.38s  ✅ ALREADY FAST (don't GPU)
─────────────────────────────────────
Total:                           0.87s
```

## Optimized Pipeline Target

```
Frame Loading (NVDEC GPU):       0.03s  (5.7x faster!)
YOLO Detection (FP16 TensorRT):  0.32s  (unchanged)
Mask Creation (CPU):             0.38s  (keep on CPU)
─────────────────────────────────────
Total:                           0.73s  (1.2x speedup, 0.14s saved)
```

## What We Built

### ✅ Completed

1. **Kornia GPU Masks** (yolo_detector.py)
   - `create_masks_batch_gpu()` method
   - Graceful fallback to CPU
   - **Status**: Disabled (too slow, but code ready for future zero-copy)

2. **NVDEC GPU Video Loader** (nvdec_video_loader.py)
   - Hardware H.264/HEVC decode
   - PyNvVideoCodec integration
   - **Status**: Ready to test & integrate

3. **INT8 TensorRT Engine** (best_int8_rtx4090.engine)
   - 7.97 MB engine file
   - **Status**: Built but buggy (CUDA errors)

### 📦 Ready to Install

```bash
pip install kornia              # ✅ Installed
pip install PyNvVideoCodec      # ✅ Installed
```

## Next Steps

### Priority 1: Integrate NVDEC

1. Test nvdec_video_loader.py with production videos
2. Integrate into server_production.py (line 1435)
3. Benchmark vs CPU decode (target: 4-8x faster)

### Priority 2: Measure Real Production Bottlenecks

1. Add detailed timing to server_production.py
2. Identify actual slow operations
3. Focus optimization there

### Future: Zero-Copy GPU Pipeline

Once NVDEC is working:
1. Keep decoded frames as CUDA tensors
2. YOLO inference on GPU tensors (modify preprocessing)
3. GPU mask creation with zero transfer
4. ProPainter on GPU tensors

**Estimated gain**: Additional 0.2-0.3s (eliminate CPU↔GPU copies)

## Lessons Learned

1. **Measure first, optimize second**: Mask creation wasn't the bottleneck
2. **Transfer overhead matters**: 600MB CPU↔GPU kills GPU gains
3. **CPU is fast for simple ops**: cv2.GaussianBlur is already optimal
4. **GPU shines with zero-copy**: Need full GPU pipeline to benefit
5. **FP16 is excellent**: INT8 complexity not worth 1.04x gain (if it worked)

## Files Modified

```
yolo_detector.py:           Added create_masks_batch_gpu() (disabled for now)
server_production.py:       GPU mask integration (falls back to CPU)
nvdec_video_loader.py:      NEW - NVDEC GPU video decoder
test_gpu_masks.py:          NEW - GPU mask benchmarking
benchmark_int8_vs_fp16.py:  NEW - TensorRT engine comparison
```

## Benchmark Scripts

```bash
# Test GPU masks (currently shows GPU is slower)
python test_gpu_masks.py

# Test NVDEC decode (needs video file)
python nvdec_video_loader.py <video.mp4>

# Test INT8 vs FP16 (INT8 crashes)
python benchmark_int8_vs_fp16.py
```

---

**Recommendation**: Focus on NVDEC GPU video decode (0.17s → 0.03s). Skip INT8 (buggy) and GPU masks (transfer overhead). FP16 YOLO at 946 fps is already excellent.
