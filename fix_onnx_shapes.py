"""
Fix ONNX graph shapes for mmdeploy::MMCVModulatedDeformConv2d operations.
The custom DCNv2 ops need explicit output shapes for TensorRT to parse correctly.
"""
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from pathlib import Path

def fix_dcnv2_shapes(onnx_path: str, output_path: str = None):
    """Fix output shapes for all DCNv2 operations in the ONNX graph."""
    if output_path is None:
        output_path = onnx_path

    print(f"Loading ONNX model from: {onnx_path}")
    model = onnx.load(onnx_path)
    graph = model.graph

    # Run shape inference first to get as much shape info as possible
    print("Running initial ONNX shape inference...")
    try:
        model = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
        print("Initial shape inference completed")
    except Exception as e:
        print(f"Initial shape inference had issues (expected for custom ops): {e}")

    # Find all DCNv2 nodes
    dcnv2_nodes = [node for node in graph.node if node.op_type == 'MMCVModulatedDeformConv2d']
    print(f"Found {len(dcnv2_nodes)} DCNv2 nodes")

    # Build a map of tensor names to their shape info
    tensor_shapes = {}
    for vi in graph.value_info:
        tensor_shapes[vi.name] = vi
    for input_vi in graph.input:
        tensor_shapes[input_vi.name] = input_vi
    for output_vi in graph.output:
        tensor_shapes[output_vi.name] = output_vi

    # Process each DCNv2 node
    for i, node in enumerate(dcnv2_nodes):
        print(f"\nProcessing DCNv2 node {i+1}/{len(dcnv2_nodes)}: {node.name}")

        # DCNv2 inputs: input, offset, mask, weight, [bias]
        if len(node.input) < 4:
            print(f"  WARNING: Expected at least 4 inputs, got {len(node.input)}")
            continue

        input_name = node.input[0]
        weight_name = node.input[3]
        output_name = node.output[0]

        print(f"  Input: {input_name}")
        print(f"  Weight: {weight_name}")
        print(f"  Output: {output_name}")

        # Get input and weight shapes
        input_shape = None
        weight_shape = None

        # Check value_info
        if input_name in tensor_shapes:
            input_shape = [d.dim_value if d.dim_value > 0 else None for d in tensor_shapes[input_name].type.tensor_type.shape.dim]
        if weight_name in tensor_shapes:
            weight_shape = [d.dim_value if d.dim_value > 0 else None for d in tensor_shapes[weight_name].type.tensor_type.shape.dim]

        # Check initializers for weight shape
        if weight_shape is None or weight_shape == [] or (weight_shape and weight_shape[0] is None):
            for init in graph.initializer:
                if init.name == weight_name:
                    weight_shape = list(init.dims)
                    break

        print(f"  Input shape: {input_shape}")
        print(f"  Weight shape: {weight_shape}")

        # For DCNv2 with stride=1, padding=1: output spatial dims = input spatial dims
        if (input_shape and weight_shape and len(input_shape) == 4 and len(weight_shape) == 4 and
            all(d is not None for d in input_shape) and weight_shape[0] is not None):
            # Output shape: [N, C_out, H, W]
            # N from input, C_out from weight, H and W from input (with stride=1, padding=1)
            output_shape = [
                input_shape[0],  # batch
                weight_shape[0],  # out_channels
                input_shape[2],  # height (preserved with stride=1, padding=1)
                input_shape[3],  # width
            ]
            print(f"  Computed output shape from inputs: {output_shape}")
        elif input_shape and len(input_shape) == 4 and all(d is not None for d in input_shape):
            # Fallback: For RFCNet, all DCNv2 layers have 128 output channels
            # This is based on the BidirectionalPropagation architecture (channel=128)
            output_shape = [
                input_shape[0],  # batch
                128,  # out_channels (RFCNet fixed architecture)
                input_shape[2],  # height
                input_shape[3],  # width
            ]
            print(f"  Using fallback shape (128 channels for RFCNet): {output_shape}")
        else:
            # Last resort: Use fixed shape from export (1x8x2x480x480 -> after ops -> 1x128xHxW)
            # Based on the default export shape
            output_shape = [1, 128, 60, 60]  # Approx after 8x downsample
            print(f"  Using hardcoded fallback shape for RFCNet: {output_shape}")

        # Create or update value_info for output
        output_value_info = helper.make_tensor_value_info(
            output_name,
            onnx.TensorProto.FLOAT,
            output_shape
        )

        # Remove existing value_info for this output if present
        value_info_to_keep = [vi for vi in graph.value_info if vi.name != output_name]
        del graph.value_info[:]
        graph.value_info.extend(value_info_to_keep)

        # Add new value_info
        graph.value_info.append(output_value_info)
        print(f"  Set output shape to: {output_shape}")

    # Run shape inference
    print("\nRunning ONNX shape inference...")
    try:
        model = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
        print("Shape inference completed")
    except Exception as e:
        print(f"Shape inference failed (this is expected for custom ops): {e}")

    # Save fixed model
    print(f"\nSaving fixed ONNX model to: {output_path}")
    onnx.save(model, output_path)
    print("Done!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_onnx_shapes.py <onnx_file> [output_file]")
        sys.exit(1)

    onnx_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    fix_dcnv2_shapes(onnx_file, output_file)
