#!/usr/bin/env python3
"""
Convert NeuFlow ONNX model from fixed batch size to dynamic batch size.

This enables batch processing of multiple frame pairs simultaneously,
eliminating the need for sequential Python loops.

Expected speedup: 10-20x (from ~3s to ~0.15-0.3s per segment)
"""

import os
import sys
import onnx
from onnx import numpy_helper
import onnxruntime as ort

def analyze_model(model_path):
    """Analyze the current ONNX model structure"""
    print(f"[INFO] Loading model: {model_path}")
    model = onnx.load(model_path)

    print("\n" + "=" * 70)
    print("MODEL ANALYSIS")
    print("=" * 70)

    # Check IR version and opset
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Opset: {model.opset_import[0].version if model.opset_import else 'unknown'}")

    # Analyze inputs
    print("\nInputs:")
    for input_tensor in model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  - {input_tensor.name}: {input_tensor.type.tensor_type.elem_type} {shape}")

    # Analyze outputs
    print("\nOutputs:")
    for output_tensor in model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  - {output_tensor.name}: {output_tensor.type.tensor_type.elem_type} {shape}")

    return model


def make_dynamic_batch(model, input_names=None):
    """
    Modify ONNX model to support dynamic batch dimension.

    Changes the first dimension of all inputs/outputs from fixed size to dynamic.
    """
    print("\n" + "=" * 70)
    print("CONVERTING TO DYNAMIC BATCH")
    print("=" * 70)

    # Make inputs dynamic
    for input_tensor in model.graph.input:
        if input_names is None or input_tensor.name in input_names:
            # Get the first dimension and make it dynamic
            dim0 = input_tensor.type.tensor_type.shape.dim[0]
            original_value = dim0.dim_value

            # Clear the fixed value and set a symbolic name
            dim0.Clear()
            dim0.dim_param = "batch_size"

            print(f"[OK] Input '{input_tensor.name}': Changed dim[0] from {original_value} to 'batch_size' (dynamic)")

    # Make outputs dynamic
    for output_tensor in model.graph.output:
        # Get the first dimension and make it dynamic
        dim0 = output_tensor.type.tensor_type.shape.dim[0]
        original_value = dim0.dim_value

        # Clear the fixed value and set a symbolic name
        dim0.Clear()
        dim0.dim_param = "batch_size"

        print(f"[OK] Output '{output_tensor.name}': Changed dim[0] from {original_value} to 'batch_size' (dynamic)")

    return model


def verify_dynamic_model(model_path):
    """Verify the dynamic model can be loaded and run with different batch sizes"""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Load with ONNX Runtime
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    print(f"\n[TEST] Testing different batch sizes...")

    import numpy as np

    # Test batch sizes: 1, 4, 16, 32
    for batch_size in [1, 4, 16, 32]:
        try:
            # Create dummy inputs
            inputs = {}
            for input_meta in session.get_inputs():
                shape = [batch_size if dim == 'batch_size' or (isinstance(dim, str) and 'batch' in dim) else dim
                        for dim in input_meta.shape]
                # Handle remaining symbolic dimensions
                shape = [batch_size if isinstance(dim, str) else dim for dim in shape]
                inputs[input_meta.name] = np.random.randn(*shape).astype(np.float32)

            # Run inference
            outputs = session.run(None, inputs)

            print(f"  [OK] Batch size {batch_size:2d}: Input {inputs[session.get_inputs()[0].name].shape} -> Output {outputs[0].shape}")
        except Exception as e:
            print(f"  [FAIL] Batch size {batch_size:2d}: FAILED - {e}")
            return False

    print("\n[OK] All batch size tests passed! Model supports dynamic batching.")
    return True


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    original_model_path = os.path.join(project_root, "models", "neuflow_things.onnx")
    output_model_path = os.path.join(project_root, "models", "neuflow_things_dynamic.onnx")

    if not os.path.exists(original_model_path):
        print(f"[ERROR] Model not found: {original_model_path}")
        print(f"[INFO] Please download NeuFlow v2 ONNX model first")
        return 1

    print("=" * 70)
    print("NEUFLOW v2 DYNAMIC BATCH CONVERTER")
    print("=" * 70)
    print(f"Input:  {original_model_path}")
    print(f"Output: {output_model_path}")
    print()

    # Step 1: Analyze original model
    model = analyze_model(original_model_path)

    # Step 2: Make batch dimension dynamic
    model = make_dynamic_batch(model)

    # Step 3: Save modified model
    print(f"\n[SAVE] Saving dynamic batch model to: {output_model_path}")
    onnx.save(model, output_model_path)
    print(f"[OK] Model saved ({os.path.getsize(output_model_path) / 1024 / 1024:.2f} MB)")

    # Step 4: Verify the model works
    if verify_dynamic_model(output_model_path):
        print("\n" + "=" * 70)
        print("SUCCESS! Dynamic batch model created and verified.")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. The new model is saved at: {output_model_path}")
        print(f"2. Update flow_comp_neuflow.py to use batched inference")
        print(f"3. Expected speedup: 10-20x faster optical flow!")
        return 0
    else:
        print("\n[ERROR] Verification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
