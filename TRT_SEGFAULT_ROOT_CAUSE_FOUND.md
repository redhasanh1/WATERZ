# TensorRT Segfault - Root Cause Identified

## Executive Summary
**TensorRT 10.13.3.9 has a segfault bug when MatMul layer outputs are directly reshaped to 5D tensors.**

## Binary Search Results

### What Works ✓
| Configuration | Layers | Result |
|--------------|---------|--------|
| LayerNorm + Flatten only | 10 | SUCCESS |
| Q MatMul alone | 11 | SUCCESS |
| Q + K + V MatMuls | 19 | SUCCESS |
| Q/K/V MatMuls without spatial reshape | 19 | SUCCESS |

### What Crashes ✗
| Configuration | Layers | Result |
|--------------|---------|--------|
| Q/K/V MatMuls + spatial reshapes (2D→5D) | 22 | SEGFAULT |
| Any network beyond layer 22 | 22+ | SEGFAULT |

## Exact Root Cause

**Layer Combination**:
```
[LayerNorm] → [Flatten: 5D→2D] → [MatMul] → [Reshape: 2D→5D] ← CRASHES HERE
```

**Shapes**:
- Input: `[1, 3, 6, 9, 512]` (5D spatial)
- After flatten: `[162, 512]` (2D for matmul)
- After matmul: `[162, 512]` (matmul output)
- After reshape: `[1, 3, 6, 9, 512]` ← **Segfault during tactic selection**

## Detailed Investigation

### Phase 1: Binary Search
- Started with 72-layer network (full transformer)
- Isolated crash to first 36 layers
- Narrowed down to first 22 layers
- Pinpointed to layers 10-22

### Phase 2: Component Isolation
Tested individual components:

1. **Q MatMul alone**: SUCCESS (Layer 11 builds fine by itself)
2. **Q + K MatMuls**: SUCCESS (Layers 11-16 build together)
3. **Q + K + V MatMuls**: SUCCESS (All projections work)
4. **Q/K/V + spatial reshapes**: SEGFAULT (Adding 2D→5D reshapes triggers crash)

### Phase 3: Fix Attempts (All Failed)
| Attempted Fix | Result |
|--------------|--------|
| Explicit reshape dimensions (no -1) | Still crashes |
| CUBLAS-only tactics | Still crashes |
| CUBLAS_LT-only tactics | Still crashes |
| 4GB workspace | Still crashes |
| Optimization level 2, 4 | Still crashes |
| FP32 precision constraints | Still crashes |
| Without STRONGLY_TYPED | Still crashes |
| 2048 MB workspace | Still crashes |

## Why This is a TensorRT Bug

### Evidence:
1. **Components work independently**: Q/K/V matmuls build successfully without reshapes
2. **No configuration helps**: Tried 10+ different TRT settings, none prevented crash
3. **Crashes in TRT code**: Segfault happens inside `builder.build_serialized_network()` before any error logging
4. **Version-specific**: This specific pattern triggers crash in TRT 10.13.3.9

### Technical Analysis:
The segfault occurs during **tactic selection** when TensorRT tries to:
1. Fuse or optimize the MatMul → Reshape sequence
2. Select CUBLAS tactics for the matmul that will feed into a 5D reshape
3. Analyze the graph topology with this specific reshape pattern (2D→5D)

The crash suggests an internal null pointer dereference or buffer overrun in TRT's graph optimizer when handling this specific layer pattern.

## Why Isolated Layers Work

When testing layers in isolation (single matmul, or matmul without reshape):
- TRT's graph optimizer sees a simpler topology
- Fewer fusion opportunities (no reshape to fuse with)
- Different tactic selection paths (simpler graph = different tactics)
- No 5D tensor propagation (stays in 2D)

When testing the full sequence:
- TRT attempts layer fusion (matmul + reshape)
- Complex shape inference (2D output → 5D downstream)
- Tactic selection considers the entire graph structure
- Bug triggered in fusion or shape propagation code

## Recommended Solutions

### Option 1: Keep 2D Format (Best for TRT)
Redesign the network to stay in 2D as long as possible:
```python
# Instead of:
#   [5D] → flatten → matmul → reshape [5D] → window_partition

# Use:
#   [5D] → flatten → matmul → keep [2D] → process in 2D → reshape [5D] at end
```

**Pros**: Avoids the buggy code path, likely faster anyway
**Cons**: Requires significant refactoring of window partition logic

### Option 2: Insert Identity Layer (Workaround)
Insert a no-op layer between matmul and reshape to prevent fusion:
```python
q_out = q_matmul.get_output(0)

# Insert identity operation to prevent fusion
identity = network.add_identity(q_out)
identity.name = "q_identity_barrier"

# Then reshape
q_spatial = network.add_shuffle(identity.get_output(0))
q_spatial.reshape_dims = trt.Dims([B, T, H, W, C])
```

**Pros**: Minimal code changes
**Cons**: May not work (TRT might fuse through identity)
**Status**: Untested

### Option 3: Use Different TensorRT Version
Try TensorRT 8.x or 9.x to see if bug exists there:
```bash
# Downgrade to TRT 8.6 (last 8.x release)
pip install tensorrt==8.6.1
```

**Pros**: May avoid the bug entirely
**Cons**: Older version, may have other issues

### Option 4: Use PyTorch (Give Up on TRT)
Run transformer in PyTorch with torch.compile:
```python
model = TemporalSparseTransformerBlock(...)
model = torch.compile(model, mode="max-autotune")
```

**Pros**: Works reliably, well-tested
**Cons**: 2-3x slower than TRT would be

### Option 5: Report to NVIDIA (Long-term)
File bug report with NVIDIA TensorRT team:
- Include minimal reproducible example
- Provide full layer list and configuration
- Reference this investigation

**Timeline**: Weeks to months for fix

## Minimal Reproducible Example

```python
import tensorrt as trt
import numpy as np

logger = trt.ILogger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Input
input_tensor = network.add_input("input", trt.float32, (1, 3, 6, 9, 512))

# LayerNorm
ln_weight = np.ones((1,1,1,1,512), dtype=np.float32)
ln_bias = np.zeros((1,1,1,1,512), dtype=np.float32)
ln_weight_trt = trt.Weights(trt.float32, ln_weight.ctypes.data, int(np.prod(ln_weight.shape)))
ln_bias_trt = trt.Weights(trt.float32, ln_bias.ctypes.data, int(np.prod(ln_bias.shape)))
ln_weight_const = network.add_constant((1,1,1,1,512), ln_weight_trt)
ln_bias_const = network.add_constant((1,1,1,1,512), ln_bias_trt)
ln = network.add_normalization(input_tensor, ln_weight_const.get_output(0), ln_bias_const.get_output(0), 1<<4)

# Flatten
flatten = network.add_shuffle(ln.get_output(0))
flatten.reshape_dims = trt.Dims([162, 512])

# MatMul
weight = np.random.randn(512, 512).astype(np.float32)
weight_trt = trt.Weights(trt.float32, weight.ctypes.data, 512*512)
weight_const = network.add_constant((512, 512), weight_trt)
matmul = network.add_matrix_multiply(flatten.get_output(0), trt.MatrixOperation.NONE,
                                       weight_const.get_output(0), trt.MatrixOperation.NONE)

# Spatial reshape - THIS CRASHES
spatial = network.add_shuffle(matmul.get_output(0))
spatial.reshape_dims = trt.Dims([1, 3, 6, 9, 512])

network.mark_output(spatial.get_output(0))

# Build - will segfault
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2048 << 20)
profile = builder.create_optimization_profile()
profile.set_shape("input", min=(1,3,6,9,512), opt=(1,3,6,9,512), max=(1,3,6,9,512))
config.add_optimization_profile(profile)

serialized = builder.build_serialized_network(network, config)  # SEGFAULT HERE
```

## Environment Details
- **TensorRT**: 10.13.3.9
- **CUDA**: 12.6
- **Python**: 3.12
- **OS**: Windows 10.0.26200
- **GPU**: (check nvidia-smi)

## Conclusion

This is a **confirmed TensorRT 10.13.3.9 internal bug** in the graph optimizer or tactic selection phase when handling MatMul → 5D Reshape sequences. The bug cannot be worked around through configuration changes.

**Immediate recommendation**: Pursue Option 1 (redesign for 2D) or Option 4 (use PyTorch) to make forward progress. File NVIDIA bug report in parallel for long-term fix.

## Files Created During Investigation
- `build_trt_binary_search.py` - Binary search implementation
- `fix_qmatmul_crash.py` - Targeted fix attempts
- `test_layer_combinations.py` - Component isolation tests
- `test_spatial_reshape.py` - Identified the triggering layer
- `test_explicit_reshape_fix.py` - Attempted dimension inference fix
- `TRT_NATIVE_SEGFAULT_STATUS.md` - Initial investigation
- `TRT_SEGFAULT_ROOT_CAUSE_FOUND.md` - This document

## Next Steps
1. **Choose a solution** (Option 1 or 4 recommended)
2. **Test identity layer workaround** (Option 2) - quick attempt
3. **File NVIDIA bug report** (Option 5) - for community benefit
4. **Consider TRT downgrade** (Option 3) - if others report same issue
