"""
Export WavePaint ONNX to TensorRT - Simple version (no pycuda needed)
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python_packages'))

print("=" * 60)
print("WavePaint TensorRT Export (Simple)")
print("=" * 60)
print("")

# Check ONNX file
onnx_path = "weights/wavepaint.onnx"
if not os.path.exists(onnx_path):
    print(f"❌ ONNX file not found: {onnx_path}")
    print("Run EXPORT_WAVEPAINT_TENSORRT.bat first to create ONNX")
    input("Press Enter to exit...")
    sys.exit(1)

print(f"✅ ONNX file found: {onnx_path}")
print("")

# Convert ONNX to TensorRT
print("[1/2] Converting ONNX to TensorRT...")
print("This may take 2-5 minutes...")
print("")

try:
    import tensorrt as trt
    import numpy as np

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
    print("")

except ImportError as e:
    print(f"❌ TensorRT import failed: {e}")
    print("")
    print("Use ONNX Runtime instead (INSTALL_ONNXRUNTIME.bat)")
    input("Press Enter to exit...")
    sys.exit(1)
except Exception as e:
    print(f"❌ TensorRT conversion failed: {e}")
    import traceback
    traceback.print_exc()
    input("Press Enter to exit...")
    sys.exit(1)

print("=" * 60)
print("✅ EXPORT COMPLETE!")
print("=" * 60)
print("")
print(f"TensorRT engine created: {engine_path}")
print("")
print("Now you can use this engine for 10-20x faster inference!")
print("Next: Create inference script to use the engine")
print("")
input("Press Enter to exit...")
