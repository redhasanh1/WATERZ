"""Minimal TensorRT engine test - just verify it loads and runs"""
import os
import sys
from pathlib import Path

# Setup paths BEFORE imports
SCRIPT_DIR = Path(__file__).parent.absolute()
TENSORRT_LIB = SCRIPT_DIR / "TensorRT-10.13.3.9" / "lib"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
os.environ['PATH'] = str(TENSORRT_LIB) + os.pathsep + CUDA_BIN + os.pathsep + os.environ.get('PATH', '')

if sys.platform == 'win32' and sys.version_info >= (3, 8):
    os.add_dll_directory(str(TENSORRT_LIB))
    os.add_dll_directory(CUDA_BIN)

import ctypes
import tensorrt as trt
import torch

print("=" * 80)
print("MINIMAL TRT ENGINE TEST")
print("=" * 80)

# Load plugin
plugin_path = "dcnv4_tensorrt_plugin/build/Release/dcnv4_plugin.dll"
plugin_dir = Path(plugin_path).parent.absolute()
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    os.add_dll_directory(str(plugin_dir))

try:
    ctypes.CDLL(str(plugin_path))
    print("[OK] Plugin loaded")
except Exception as e:
    print(f"[FAIL] Plugin load failed: {e}")
    sys.exit(1)

# Initialize TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)  # Only show errors
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
print("[OK] TensorRT initialized")

# Load engine
engine_path = "engines/rfcnet/rfcnet_dcnv4_fp16.engine"
runtime = trt.Runtime(TRT_LOGGER)

with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

if not engine:
    print("[FAIL] Engine deserialization failed")
    sys.exit(1)

print(f"[OK] Engine loaded ({Path(engine_path).stat().st_size // (1024*1024)} MB)")

# Create context
context = engine.create_execution_context()
print("[OK] Execution context created")

# Test inference
B, T, H, W = 1, 8, 256, 256
context.set_input_shape("masked_flows", (B, T, 2, H, W))
context.set_input_shape("masks", (B, T, 1, H, W))

masked_flows = torch.randn(B, T, 2, H, W).cuda()
masks = torch.ones(B, T, 1, H, W).cuda()
output = torch.zeros(B, T, 2, H, W).cuda()

bindings = [masked_flows.data_ptr(), masks.data_ptr(), output.data_ptr()]
stream = torch.cuda.current_stream().cuda_stream

try:
    success = context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    if success:
        print(f"[OK] Inference executed successfully")
        print(f"[OK] Output shape: {output.shape}")
        print(f"[OK] Output range: [{output.min():.3f}, {output.max():.3f}]")
        print()
        print("=" * 80)
        print("[SUCCESS] TensorRT engine is working!")
        print("=" * 80)
    else:
        print("[FAIL] Execution returned False")
        sys.exit(1)

except Exception as e:
    print(f"[FAIL] Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
