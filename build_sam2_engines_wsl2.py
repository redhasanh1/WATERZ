"""
Build SAM2 TensorRT engines for WSL2 (Linux)
Uses Python TensorRT API to convert ONNX → TensorRT engines
"""
import tensorrt as trt
import os

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Paths
ONNX_DIR = "/mnt/d/watermarkz/sam2_trt_inference/sam2_pytorch2onnx/output"
ENGINE_DIR = "/mnt/d/watermarkz/sam2_trt_inference/engines_wsl2"  # NEW directory for WSL2 engines

# ONNX models
ENCODER_ONNX = f"{ONNX_DIR}/sam2.1_hiera_tiny_encoder.onnx"
DECODER_ONNX = f"{ONNX_DIR}/sam2.1_hiera_tiny_decoder.onnx"

# Output engines
ENCODER_ENGINE = f"{ENGINE_DIR}/sam2_encoder_fp16.engine"
DECODER_ENGINE = f"{ENGINE_DIR}/sam2_decoder_fp16_dynamic.engine"

def build_engine(onnx_path, engine_path, fp16=True, dynamic_shapes=None):
    """
    Build TensorRT engine from ONNX model

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        fp16: Enable FP16 precision
        dynamic_shapes: Dict of {input_name: [(min, opt, max), ...]}
    """
    print(f"\n{'='*70}")
    print(f"Building: {os.path.basename(engine_path)}")
    print(f"{'='*70}")
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"FP16: {fp16}")
    print(f"Dynamic shapes: {dynamic_shapes}")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"\n[1/4] Parsing ONNX...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    print(f"✓ Parsed {network.num_inputs} inputs, {network.num_outputs} outputs")

    # Configure builder
    print(f"\n[2/4] Configuring builder...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 enabled")

    # Set optimization level
    config.builder_optimization_level = 5
    print("✓ Optimization level: 5")

    # Set dynamic shapes if provided
    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for input_name, shapes in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = shapes
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            print(f"✓ Dynamic shape '{input_name}':")
            print(f"    Min: {min_shape}")
            print(f"    Opt: {opt_shape}")
            print(f"    Max: {max_shape}")
        config.add_optimization_profile(profile)

    # Build engine
    print(f"\n[3/4] Building engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False

    # Save engine
    print(f"\n[4/4] Saving engine...")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"✓ Saved: {engine_path} ({size_mb:.2f} MB)")
    print(f"{'='*70}\n")

    return True

def main():
    print("="*70)
    print("SAM2 TensorRT Engine Builder for WSL2")
    print("="*70)
    print()
    print("This will build TensorRT engines optimized for WSL2 (Linux)")
    print(f"Output directory: {ENGINE_DIR}")
    print()

    # Build encoder (dynamic batch size)
    print("\n[ENCODER] Building encoder engine with dynamic batch size...")
    encoder_dynamic_shapes = {
        'image': [
            (1, 3, 1024, 1024),  # min: batch=1
            (1, 3, 1024, 1024),  # opt: batch=1
            (1, 3, 1024, 1024),  # max: batch=1
        ],
    }

    success = build_engine(
        ENCODER_ONNX,
        ENCODER_ENGINE,
        fp16=True,
        dynamic_shapes=encoder_dynamic_shapes
    )

    if not success:
        print("❌ Failed to build encoder")
        return

    # Build decoder (dynamic shapes for mask prompts)
    print("\n[DECODER] Building decoder engine with dynamic shapes...")

    # Decoder has dynamic inputs for num_labels (usually 1)
    # Based on actual ONNX shapes
    decoder_dynamic_shapes = {
        'image_embed': [
            (1, 256, 64, 64),  # fixed
            (1, 256, 64, 64),
            (1, 256, 64, 64),
        ],
        'high_res_feats_0': [
            (1, 32, 256, 256),  # fixed
            (1, 32, 256, 256),
            (1, 32, 256, 256),
        ],
        'high_res_feats_1': [
            (1, 64, 128, 128),  # fixed
            (1, 64, 128, 128),
            (1, 64, 128, 128),
        ],
        'point_coords': [
            (1, 2, 2),   # min: num_labels=1
            (1, 2, 2),   # opt: num_labels=1
            (10, 2, 2),  # max: num_labels=10
        ],
        'point_labels': [
            (1, 2),   # min
            (1, 2),   # opt
            (10, 2),  # max
        ],
        'mask_input': [
            (1, 1, 256, 256),   # min
            (1, 1, 256, 256),   # opt
            (10, 1, 256, 256),  # max
        ],
        'has_mask_input': [
            (1,),  # fixed
            (1,),
            (1,),
        ],
    }

    success = build_engine(
        DECODER_ONNX,
        DECODER_ENGINE,
        fp16=True,
        dynamic_shapes=decoder_dynamic_shapes
    )

    if not success:
        print("❌ Failed to build decoder")
        return

    print("\n" + "="*70)
    print("✅ SUCCESS! Both engines built for WSL2")
    print("="*70)
    print()
    print("Engines saved to:")
    print(f"  - {ENCODER_ENGINE}")
    print(f"  - {DECODER_ENGINE}")
    print()
    print("Next step: Update test_sam2_wsl2.py to use these engines")
    print("="*70)

if __name__ == "__main__":
    main()
