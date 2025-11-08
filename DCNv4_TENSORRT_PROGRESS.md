# DCNv4 TensorRT Plugin - Development Progress

## Overview

Complete implementation of DCNv4 TensorRT plugin for RFC Net acceleration, targeting **6.5-9x total speedup** vs PyTorch baseline through combination of:
- DCNv4 deformable convolution (3x faster than DCNv2)
- TensorRT FP16 optimization (1.5-2x speedup)
- FP8 quantization for standard Conv layers (1.3-1.5x speedup)

**Hardware Target**: NVIDIA RTX 4090 (Ada Lovelace, Compute Capability 8.9)

---

## Progress Summary

### Phase 1A: DCNv4 PyTorch Validation ✅ COMPLETE
**Status**: Validated DCNv4 provides 1.3x overall speedup (3x on deformable conv operations)

**Files**:
- `faster-propainter-main/model/recurrent_flow_completion.py` - DCNv4 integration
- `test_rfcnet_dcnv4.py` - Functional test
- `benchmark_rfcnet_dcnv4.py` - Performance benchmark

**Results**:
```
Resolution   | PyTorch DCNv2 | PyTorch DCNv4 | Speedup
640x480      | 21.1 ms       | 16.2 ms       | 1.30x
1280x720     | 62.1 ms       | 48.9 ms       | 1.27x
```

**Key Finding**: DCNv4 is 3x faster on deformable conv portion (~30% of total compute), yielding 1.3x overall speedup

---

### Phase 2A-2C: Plugin Infrastructure ✅ COMPLETE
**Status**: Plugin structure, CUDA kernels (stub), and build system ready

**Files Created**:
1. **dcnv4_tensorrt_plugin/include/DCNv4Plugin.h** (253 lines)
   - Full IPluginV2DynamicExt interface
   - Parameter serialization support
   - Dynamic shape handling

2. **dcnv4_tensorrt_plugin/src/DCNv4Plugin.cpp** (580 lines)
   - Plugin implementation with all interface methods
   - Workspace calculation
   - Format conversion support ([N,C,H,W] ↔ [N,L,C])

3. **dcnv4_tensorrt_plugin/src/DCNv4PluginKernel.cu** (526 lines)
   - DCNv4 CUDA kernel wrappers
   - Format conversion kernels (NCHW ↔ NHWC)
   - FP16/FP32 template dispatch
   - Entry point: `dcnv4_forward_cuda()`

4. **dcnv4_tensorrt_plugin/src/common.h** (101 lines)
   - Bilinear interpolation utilities
   - Common CUDA helpers

5. **dcnv4_tensorrt_plugin/CMakeLists.txt** (164 lines)
   - Full CMake build system
   - CUDA 12.x support
   - TensorRT 10.13.3.9 integration
   - Multi-architecture support (SM 75, 80, 89)

6. **dcnv4_tensorrt_plugin/build.bat**
   - Windows build script

**Plugin Architecture**:
```
TensorRT Input [N,C,H,W] FP16
    ↓
DCNv4Plugin::enqueue()
    ├─ Convert NCHW → NHWC (workspace)
    ├─ dcnv4_forward_cuda() (3x faster than DCNv2)
    └─ Convert NHWC → NCHW (workspace)
    ↓
TensorRT Output [N,C,H,W] FP16
```

**Build Command**:
```bash
cd dcnv4_tensorrt_plugin
build.bat  # Windows
```

**Expected Output**: `dcnv4_plugin.dll` (Windows) or `libdcnv4_plugin.so` (Linux)

---

### Phase 2D: ONNX Export ✅ COMPLETE
**Status**: ONNX export script with DCNv4 custom op registration

**Files Created**:
1. **export_rfcnet_dcnv4_onnx.py** (390 lines)
   - Registers DCNv4 as custom ONNX op
   - Maps `DCNv4::forward` → `custom::DCNv4Plugin`
   - Handles format conversion in symbolic function
   - Dynamic shape support

2. **test_dcnv4_onnx_export.bat**
   - Test script for ONNX export

**Usage**:
```bash
python export_rfcnet_dcnv4_onnx.py \
    --weights weights/recurrent_flow_completion.pth \
    --out engines/rfcnet/rfcnet_dcnv4.onnx \
    --optshape 1x8x2x640x480
```

**Key Features**:
- Automatic DCNv4 detection and enabling (`ENABLE_DCNV4_RFCNET=1`)
- Custom symbolic function for `DCNv4::forward`
- Dynamic axes for batch/time/spatial dimensions
- Verification of DCNv4 layers in model

**Expected Output**: `engines/rfcnet/rfcnet_dcnv4.onnx` (~50-100 MB)

---

### Phase 2E: TensorRT Engine Builder ✅ COMPLETE
**Status**: Complete engine builder with DCNv4 plugin loading

**Files Created**:
1. **build_rfcnet_trt_engine.py** (380 lines)
   - DCNv4 plugin loading via `ctypes.CDLL()`
   - Plugin registry verification
   - ONNX parsing with custom ops
   - Dynamic shape optimization profiles
   - FP16 precision configuration

2. **build_rfcnet_dcnv4.bat**
   - Build script with automatic checks

**Usage**:
```bash
python build_rfcnet_trt_engine.py \
    --onnx engines/rfcnet/rfcnet_dcnv4.onnx \
    --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine \
    --fp16 \
    --min-shape 1x8x2x256x256 \
    --opt-shape 1x8x2x640x480 \
    --max-shape 1x16x2x1280x720
```

**Optimization Profiles**:
```
Min:  1 batch, 8 frames,  256x256 (small)
Opt:  1 batch, 8 frames,  640x480 (SD)
Max:  1 batch, 16 frames, 1280x720 (HD)
```

**Key Features**:
- Plugin loading and registry verification
- FP16 precision (1.5-2x speedup)
- 4GB workspace for complex ops
- All tactic sources (cuBLAS, cuBLASLt, cuDNN)
- Maximum optimization level (5)

**Expected Output**: `engines/rfcnet/rfcnet_dcnv4_fp16.engine` (~50-100 MB)

---

### Phase 2F: Testing Suite ✅ COMPLETE
**Status**: Comprehensive testing with 5 test categories

**Files Created**:
1. **test_rfcnet_trt_dcnv4.py** (500+ lines)
   - Test 1: Plugin loading
   - Test 2: Engine deserialization
   - Test 3: Inference execution
   - Test 4: Numerical accuracy (vs PyTorch)
   - Test 5: Performance benchmarking

**Usage**:
```bash
python test_rfcnet_trt_dcnv4.py \
    --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine \
    --iterations 100
```

**Test Coverage**:
1. **Plugin Loading Test**:
   - Verifies DCNv4 plugin DLL loads
   - Checks plugin registration in TensorRT registry
   - Validates plugin creator availability

2. **Engine Loading Test**:
   - Deserializes TensorRT engine
   - Creates execution context
   - Verifies I/O tensor configurations

3. **Inference Test**:
   - Allocates GPU buffers
   - Executes inference with dummy data
   - Validates output shapes

4. **Accuracy Test**:
   - Compares TensorRT vs PyTorch outputs
   - Calculates max/mean absolute difference
   - Calculates relative error percentage
   - **Pass Criteria**: max diff < 0.1, relative < 5%

5. **Performance Test**:
   - Warmup iterations (10% of total)
   - Timed iterations with CUDA synchronization
   - Statistics: mean, std, P50, P95, P99, FPS

**Expected Performance** (640x480):
```
PyTorch DCNv4: ~16 ms
Target TensorRT: 7-10 ms (1.6-2.3x speedup)
```

---

### Phase 2G: CUDA Kernel Implementation ⏳ PENDING
**Status**: Placeholder implementation ready for actual CUDA kernels

**Required Work**:
1. **Complete DCNv4PluginKernel.cu**:
   - Integrate actual DCNv4 CUDA kernels from `DCNv4/DCNv4_op/src/cuda/`
   - Implement efficient NCHW ↔ NLC format conversion
   - Optimize for Ada Lovelace (SM 8.9)
   - Add FP16 template specializations

2. **Optimize DCNv4Plugin.cpp**:
   - Refine workspace size calculation
   - Add detailed error handling and logging
   - Profile and optimize `enqueue()` method

3. **Production Integration**:
   - Update `server_production.py` to use TensorRT engine
   - Modify `START_CELERY_TRT.bat` environment variables
   - Test end-to-end watermark removal pipeline

**Current Status**: Stub implementation (passthrough)
```cpp
// Phase 2G TODO: Implement full forward pass
// For now, just copy input to output as passthrough
size_t bytes = batch * channels * height * width * sizeof(__half);
cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice, stream);
```

---

## File Inventory

### Core Plugin Files (7 files)
```
dcnv4_tensorrt_plugin/
├── include/
│   └── DCNv4Plugin.h                  # 253 lines - Plugin interface
├── src/
│   ├── DCNv4Plugin.cpp                # 580 lines - Plugin implementation
│   ├── DCNv4PluginKernel.cu           # 526 lines - CUDA kernels (stub)
│   └── common.h                       # 101 lines - CUDA utilities
├── CMakeLists.txt                     # 164 lines - Build system
├── build.bat                          # Windows build script
└── README.md                          # 269 lines - Documentation
```

### ONNX Export (2 files)
```
export_rfcnet_dcnv4_onnx.py           # 390 lines - ONNX export with custom op
test_dcnv4_onnx_export.bat            # Export test script
```

### Engine Builder (2 files)
```
build_rfcnet_trt_engine.py            # 380 lines - TensorRT engine builder
build_rfcnet_dcnv4.bat                # Build script
```

### Testing (2 files)
```
test_rfcnet_trt_dcnv4.py              # 500+ lines - Comprehensive test suite
test_dcnv4_trace.py                   # 150 lines - Diagnostic script
```

**Total**: ~3,300 lines of production code

---

## Build & Test Workflow

### Step 1: Build DCNv4 Plugin
```bash
cd dcnv4_tensorrt_plugin
build.bat

# Expected output:
# dcnv4_tensorrt_plugin/build/Release/dcnv4_plugin.dll
```

### Step 2: Export ONNX Model
```bash
python export_rfcnet_dcnv4_onnx.py \
    --weights weights/recurrent_flow_completion.pth \
    --out engines/rfcnet/rfcnet_dcnv4.onnx \
    --optshape 1x8x2x640x480

# Expected output:
# engines/rfcnet/rfcnet_dcnv4.onnx (~50-100 MB)
```

### Step 3: Build TensorRT Engine
```bash
python build_rfcnet_trt_engine.py \
    --onnx engines/rfcnet/rfcnet_dcnv4.onnx \
    --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine \
    --fp16

# Expected output:
# engines/rfcnet/rfcnet_dcnv4_fp16.engine (~50-100 MB)
```

### Step 4: Test Engine
```bash
python test_rfcnet_trt_dcnv4.py \
    --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine \
    --iterations 100

# Expected results:
# [PASS] Plugin loading
# [PASS] Engine loading
# [PASS] Inference execution
# [PASS] Numerical accuracy (max diff < 0.1)
# [PASS] Performance (7-10ms @ 640x480)
```

---

## Performance Targets

### Phase 1A: PyTorch DCNv4 Validation
```
Resolution   | Baseline (DCNv2) | DCNv4    | Speedup
640x480      | 21.1 ms          | 16.2 ms  | 1.30x ✅
1280x720     | 62.1 ms          | 48.9 ms  | 1.27x ✅
```

### Phase 2: TensorRT Plugin (Target)
```
Resolution   | PyTorch DCNv4 | TensorRT FP16 | Speedup
640x480      | 16.2 ms       | 7-10 ms       | 1.6-2.3x
1280x720     | 48.9 ms       | 20-31 ms      | 1.6-2.4x
```

### Phase 3: Combined (Target)
```
Component                        | Speedup
DCNv4 deformable conv (plugin)   | 3x vs DCNv2
TensorRT FP16 standard Conv      | 1.5-2x
FP8 quantization (other layers)  | 1.3-1.5x
TOTAL (Phase 3)                  | 6.5-9x ⚡
```

---

## Known Issues & Limitations

### Phase 2G Blockers (Current)
1. **CUDA Kernel Stub**: DCNv4PluginKernel.cu uses passthrough instead of actual DCNv4 kernels
   - **Impact**: Plugin builds but doesn't provide DCNv4 functionality
   - **Fix**: Integrate actual kernels from `DCNv4/DCNv4_op/src/cuda/`

2. **Format Conversion**: NCHW ↔ NLC conversion needs optimization
   - **Impact**: Overhead from format conversion may reduce speedup
   - **Fix**: Fuse conversion with DCNv4 operations

3. **Offset Generation**: DCNv4 expects internal offset generation
   - **Impact**: Plugin needs to handle offset computation
   - **Fix**: Integrate offset generation from DCNv4 module

### Compatibility Notes
1. **TensorRT Version**: Requires TensorRT 10.13.3.9 or later
2. **CUDA Version**: Requires CUDA 12.x for Ada Lovelace support
3. **Compute Capability**: Optimized for SM 8.9 (RTX 4090), also supports SM 75 (Turing), SM 80 (Ampere)
4. **FP16 Precision**: Some numerical precision loss expected (< 5% relative error)

---

## Next Steps

### Immediate (Phase 2G)
1. **Complete CUDA kernel implementation** in DCNv4PluginKernel.cu:
   - Copy DCNv4 kernels from `DCNv4/DCNv4_op/src/cuda/dcnv4_cuda.cu`
   - Implement format conversion kernels
   - Add FP16 optimizations for Ada Lovelace

2. **Test plugin functionality**:
   - Build plugin with actual kernels
   - Run test suite: `test_rfcnet_trt_dcnv4.py`
   - Verify numerical accuracy vs PyTorch
   - Benchmark performance (target: 7-10ms @ 640x480)

3. **Production integration**:
   - Update `server_production.py` to load TensorRT engine
   - Modify inference pipeline to use TensorRT context
   - Test end-to-end watermark removal

### Future (Phase 3)
1. **FP8 Quantization**: Integrate FP8 for standard Conv layers
2. **Multi-stream Optimization**: Support concurrent inference
3. **Batch Processing**: Optimize for batch > 1
4. **Engine Caching**: Cache compiled engines per resolution

---

## Development Timeline

- **Phase 1A**: DCNv4 PyTorch validation ✅ (2 days)
- **Phase 2A-2C**: Plugin infrastructure ✅ (3 days)
- **Phase 2D**: ONNX export ✅ (1 day)
- **Phase 2E**: Engine builder ✅ (1 day)
- **Phase 2F**: Testing suite ✅ (1 day)
- **Phase 2G**: CUDA kernels & production ⏳ (2-3 days estimated)
- **Phase 3**: FP8 integration (3-5 days estimated)

**Total Progress**: ~75% complete (Phases 1A-2F done, Phase 2G pending)

---

## Technical References

- [DCNv4 Paper](https://arxiv.org/abs/2401.06197): "Efficient Deformable ConvNets"
- [DCNv4 GitHub](https://github.com/OpenGVLab/DCNv4)
- [TensorRT Plugin Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#add_custom_layer)
- [RFC Net Paper](https://arxiv.org/abs/2205.12361): "ProPainter: Improving Propagation and Transformer for Video Inpainting"

---

## Contact & Support

For issues or questions:
1. Check `dcnv4_tensorrt_plugin/README.md` for detailed plugin documentation
2. Review test output from `test_rfcnet_trt_dcnv4.py` for diagnostics
3. Enable verbose logging: `--verbose` flag in builder/tester scripts

---

*Last Updated: 2025 (Phase 2F Complete)*
