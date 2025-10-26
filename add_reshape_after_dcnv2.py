"""
Add explicit Reshape nodes after DCNv2 operations to force correct output shapes.
This works around the mmdeploy plugin's incorrect getOutputDimensions() implementation.
"""
import onnx
import onnx.helper as helper
import numpy as np
from pathlib import Path

def add_reshape_after_dcnv2(onnx_path: str, output_path: str = None):
    """Add Reshape nodes after all DCNv2 operations."""
    if output_path is None:
        output_path = onnx_path

    print(f"Loading ONNX model from: {onnx_path}")
    model = onnx.load(onnx_path)
    graph = model.graph

    # Find all DCNv2 nodes
    dcnv2_nodes = [node for node in graph.node if node.op_type == 'MMCVModulatedDeformConv2d']
    print(f"Found {len(dcnv2_nodes)} DCNv2 nodes")

    # Collect new nodes to add and modifications
    new_initializers = []
    new_nodes = []
    output_renames = {}  # Map old output name -> new reshape output name

    for i, dcnv2_node in enumerate(dcnv2_nodes):
        original_output = dcnv2_node.output[0]
        reshape_output = f"{original_output}_reshaped"

        # Create new intermediate output for DCNv2
        dcnv2_intermediate = f"{original_output}_before_reshape"

        # Update DCNv2 node to output to intermediate
        dcnv2_node.output[0] = dcnv2_intermediate

        # Create shape constant: [1, 128, 60, 60]
        shape_name = f"reshape_shape_{i}"
        shape_tensor = helper.make_tensor(
            name=shape_name,
            data_type=onnx.TensorProto.INT64,
            dims=[4],
            vals=[1, 128, 60, 60]
        )
        new_initializers.append(shape_tensor)

        # Create Reshape node
        reshape_node = helper.make_node(
            'Reshape',
            inputs=[dcnv2_intermediate, shape_name],
            outputs=[reshape_output],
            name=f"/model/feat_prop_module/reshape_dcnv2_{i}"
        )
        new_nodes.append(reshape_node)

        # Track that this output was renamed
        output_renames[original_output] = reshape_output

        print(f"  Added Reshape after {dcnv2_node.name}: {dcnv2_intermediate} -> {reshape_output}")

    # Update all nodes that consume DCNv2 outputs
    for node in graph.node:
        if node.op_type == 'MMCVModulatedDeformConv2d':
            continue  # Already modified
        for j, input_name in enumerate(node.input):
            if input_name in output_renames:
                node.input[j] = output_renames[input_name]
                print(f"  Updated {node.name} input: {input_name} -> {output_renames[input_name]}")

    # Add new initializers and nodes to graph
    graph.initializer.extend(new_initializers)
    graph.node.extend(new_nodes)

    print(f"\nAdded {len(new_nodes)} Reshape nodes and {len(new_initializers)} shape initializers")

    # Save modified model
    print(f"Saving modified ONNX model to: {output_path}")
    onnx.save(model, output_path)
    print("Done!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python add_reshape_after_dcnv2.py <onnx_file> [output_file]")
        sys.exit(1)

    onnx_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    add_reshape_after_dcnv2(onnx_file, output_file)
