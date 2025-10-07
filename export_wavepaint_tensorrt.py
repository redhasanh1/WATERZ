"""
Export WavePaint to TensorRT for 10-20x speedup
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wavepaint'))

import torch
import torch.onnx
from model import WavePaint

print("=" * 60)
print("WavePaint TensorRT Export")
print("=" * 60)
print("")

# Check CUDA
if not torch.cuda.is_available():
    print("❌ CUDA not available! TensorRT requires GPU.")
    input("Press Enter to exit...")
    sys.exit(1)

device = torch.device("cuda")
print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
print("")

# Find weights
print("[1/6] Finding WavePaint weights...")
weights = [
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_thick_mask.pth", "CelebHQ Thick - Same as test_wavepaint_only!", 2, 4),
    ("weights/WavePaint_blocks4_dim128_modules8_celebhq_medium_mask.pth", "CelebHQ Medium", 2, 4),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_thick_mask.pth", "Places256 Thick", 4, 3),
    ("weights/WavePaint__blocks4_dim128_modules8_places256_medium_mask.pth", "Places256 Medium", 4, 3),
]

weights_path = None
mult_value = 2
input_channels = 4
for path, name, mult, in_ch in weights:
    if os.path.exists(path):
        weights_path = path
        mult_value = mult
        input_channels = in_ch
        print(f"✅ Using: {name}")
        print(f"   Path: {path}")
        print(f"   Architecture: mult={mult}, input_channels={in_ch}")
        break

if not weights_path:
    print("❌ No weights found!")
    input("Press Enter to exit...")
    sys.exit(1)
print("")

# Load PyTorch model
print("[2/6] Loading PyTorch model...")
try:
    model = WavePaint(
        num_modules=8,
        blocks_per_module=4,
        mult=mult_value,
        ff_channel=128,
        final_dim=128,
        dropout=0.5
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    input("Press Enter to exit...")
    sys.exit(1)
print("")

# Create dummy inputs
print("[3/6] Creating dummy inputs...")
dummy_img = torch.randn(1, 3, 256, 256).to(device)
dummy_mask = torch.randn(1, 1, 256, 256).to(device)
print("✅ Dummy inputs created (1x3x256x256 + 1x1x256x256)")
print("")

# Export to ONNX
print("[4/6] Exporting to ONNX...")
onnx_path = "weights/wavepaint.onnx"
try:
    torch.onnx.export(
        model,
        (dummy_img, dummy_mask),
        onnx_path,
        input_names=['image', 'mask'],
        output_names=['output'],
        dynamic_axes={
            'image': {0: 'batch'},
            'mask': {0: 'batch'},
            'output': {0: 'batch'}
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"✅ ONNX exported: {onnx_path}")
except Exception as e:
    print(f"❌ ONNX export failed: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)
print("")

# Convert ONNX to TensorRT
print("[5/6] Converting ONNX to TensorRT...")
print("This may take 2-5 minutes...")
print("")

try:
    import tensorrt as trt
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoinit

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("   Parsing ONNX...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("❌ ONNX parsing failed:")
            for error in range(parser.num_errors):
                print(f"   {parser.get_error(error)}")
            input("Press Enter to exit...")
            sys.exit(1)

    # Build engine
    print("   Building TensorRT engine...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Enable FP16 for 2x speedup
    if builder.platform_has_fast_fp16:
        print("   ✅ Enabling FP16 mode (2x faster)")
        config.set_flag(trt.BuilderFlag.FP16)

    # Add optimization profile for dynamic batch size
    profile = builder.create_optimization_profile()
    profile.set_shape("image", (1, 3, 256, 256), (1, 3, 256, 256), (1, 3, 256, 256))
    profile.set_shape("mask", (1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256))
    config.add_optimization_profile(profile)
    print("   ✅ Optimization profile added")

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("❌ Engine build failed")
        input("Press Enter to exit...")
        sys.exit(1)

    # Save engine
    engine_path = "weights/wavepaint.engine"
    with open(engine_path, 'wb') as f:
        engine_bytes = bytes(serialized_engine)
        f.write(engine_bytes)

    print(f"✅ TensorRT engine saved: {engine_path}")
    print(f"   Size: {len(engine_bytes) / (1024*1024):.1f} MB")

except ImportError as e:
    print(f"❌ TensorRT import failed: {e}")
    print("")
    import traceback
    traceback.print_exc()
    print("")
    print("TensorRT is required for GPU acceleration.")
    print("It should already be installed if YOLO TensorRT works.")
    input("Press Enter to exit...")
    sys.exit(1)
except Exception as e:
    print(f"❌ TensorRT conversion failed: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)
print("")

# Test inference speed
print("[6/6] Testing speed...")
try:
    # PyTorch speed
    print("   Testing PyTorch speed...")
    import time

    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(dummy_img, dummy_mask)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = model(dummy_img, dummy_mask)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 10

    print(f"   PyTorch: {pytorch_time*1000:.0f} ms/frame ({1/pytorch_time:.2f} fps)")

    # TensorRT speed
    print("   Testing TensorRT speed...")

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    h_input_img = cuda.pagelocked_empty(trt.volume((1, 3, 256, 256)), dtype=np.float32)
    h_input_mask = cuda.pagelocked_empty(trt.volume((1, 1, 256, 256)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume((1, 3, 256, 256)), dtype=np.float32)

    d_input_img = cuda.mem_alloc(h_input_img.nbytes)
    d_input_mask = cuda.mem_alloc(h_input_mask.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # Warmup
    for _ in range(3):
        cuda.memcpy_htod_async(d_input_img, h_input_img, stream)
        cuda.memcpy_htod_async(d_input_mask, h_input_mask, stream)
        context.execute_async_v2([int(d_input_img), int(d_input_mask), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(10):
        cuda.memcpy_htod_async(d_input_img, h_input_img, stream)
        cuda.memcpy_htod_async(d_input_mask, h_input_mask, stream)
        context.execute_async_v2([int(d_input_img), int(d_input_mask), int(d_output)], stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    tensorrt_time = (time.time() - start) / 10

    print(f"   TensorRT: {tensorrt_time*1000:.0f} ms/frame ({1/tensorrt_time:.2f} fps)")
    print("")
    print(f"   🚀 Speedup: {pytorch_time/tensorrt_time:.1f}x faster!")

except Exception as e:
    print(f"   ⚠️  Could not benchmark: {e}")
print("")

print("=" * 60)
print("✅ EXPORT COMPLETE!")
print("=" * 60)
print("")
print(f"Files created:")
print(f"  - {onnx_path}")
print(f"  - {engine_path}")
print("")
print("Next: Run TEST_WAVEPAINT_TENSORRT.bat to test with real images!")
print("")
input("Press Enter to exit...")
