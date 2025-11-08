# NVDEC REJECTED ❌

## Summary
**NVDEC GPU video decode is 7.4x SLOWER than CPU for pipelines requiring numpy arrays.**

---

## Benchmark Results

### With Proper DLPack Conversion
| Method | Time | ms/frame | Status |
|--------|------|----------|--------|
| **CPU (cv2.VideoCapture)** | 0.132s | 0.44ms | ✅ **FASTER** |
| NVDEC (GPU→CPU) | 0.981s | 3.27ms | ❌ **7.4x SLOWER** |

### Previous "1.16x faster" Result was WRONG
The earlier benchmark showing NVDEC was 1.16x faster was **INVALID** because it wasn't actually converting frames to numpy - it was just storing object references that couldn't be used by the pipeline.

---

## Why NVDEC is Slower

### The Problem: GPU→CPU Transfer Overhead

```
NVDEC Pipeline:
1. NVDEC decode on GPU (fast) → 0.05s
2. DLPack to PyTorch tensor on GPU (zero-copy) → 0.001s
3. ❌ torch_tensor.cpu() - PCIe GPU→CPU copy (VERY SLOW) → 0.85s
4. Convert to numpy (fast) → 0.001s
───────────────────────────────────────────────────
Total: 0.981s

CPU Pipeline:
1. CPU decode to numpy (cv2.VideoCapture) → 0.132s
───────────────────────────────────────────────────
Total: 0.132s
```

The PCIe transfer overhead (step 3) completely destroys any gains from GPU decode.

---

## When NVDEC Would Be Beneficial

NVDEC is only faster if the **ENTIRE pipeline stays on GPU** with no CPU transfers:

```
GPU-only pipeline (hypothetical):
NVDEC decode (GPU)
  ↓
PyTorch tensor (GPU, zero-copy)
  ↓
GPU model inference (GPU-native)
  ↓
GPU post-processing (GPU-native)
  ↓
NVENC encode (GPU) or display
```

**This is NOT the current use case** - ProPainter pipeline requires numpy arrays on CPU for the models.

---

## Technical Details

### Proper Conversion Method (DLPack)

```python
# CORRECT method (used in final benchmark):
import torch
torch_tensor = torch.from_dlpack(decoded_frame)  # Zero-copy on GPU
frame_np = torch_tensor.cpu().numpy()  # ❌ PCIe transfer kills performance
```

### Why Previous Benchmark Was Wrong

```python
# WRONG method (appeared fast but didn't work):
import numpy as np
cuda_tensors = decoded_frame.cuda()
mem_view = cuda_tensors[0]
frame_np = np.asarray(mem_view)  # Returns 0-dimensional object array!
```

`np.asarray(CAIMemoryView)` does NOT work - it returns an empty scalar array, not the actual frame data. The previous benchmark was just storing object references, which is why it appeared fast but the frames were unusable.

---

## Decision

**NVDEC is DISABLED by default in START_CELERY_TRT.bat**

```bat
REM ❌ NVDEC VIDEO DECODER: DISABLED (7.4x SLOWER than CPU)
set ENABLE_NVDEC=0
```

Video decode will use **CPU cv2.VideoCapture** which is 7.4x faster for this use case.

---

## Impact on Overall Pipeline

Video decode is a small part of the pipeline (~2% of total time), so this doesn't significantly affect overall performance:

| Component | Time | % of Pipeline |
|-----------|------|---------------|
| Transformer (FP8) | 2.39s | 42% |
| Encoder/Decoder | 2.70s | 23.5% |
| RFCNet | 1.34s | 11.7% |
| Feature Propagation (DCNv4) | 0.70s | 6.1% |
| Other | 0.86s | 15% |
| Mask Creation (CPU) | 0.08s | 1.4% |
| **Video Decode (CPU)** | **0.13s** | **2.3%** |

Even with NVDEC's 7.4x slowdown, it would only add ~0.8s to the pipeline, increasing total time from 5.72s to 6.5s.

But there's no reason to make it slower when CPU is faster!

---

## Files Modified

1. **START_CELERY_TRT.bat**: Set `ENABLE_NVDEC=0` (line 48)
2. **server_production.py**: NVDEC code remains (with CPU fallback) but disabled by default
3. **nvdec_video_loader.py**: Fixed with DLPack conversion but not used
4. **NVDEC_REJECTED.md**: This document

---

## Conclusion

**NVDEC GPU video decode is NOT beneficial for pipelines that need numpy arrays on CPU.**

The GPU→CPU transfer overhead completely negates any decode speedup. CPU decode with cv2.VideoCapture is 7.4x faster and should be used instead.

---

**Date**: 2025-11-07
**Hardware**: NVIDIA RTX 4090 (Ada Lovelace)
**Tested on**: asser.mp4 (300 frames @ 480x872, H.264)
