"""
SAM2 Memory Encoder - TensorRT Wrapper

Provides a TensorRT-based implementation of the SAM2 memory encoder module.
This replaces the PyTorch memory encoder for faster video tracking.
"""

import os
import sys

# Add TensorRT DLLs to PATH (Windows-specific fix)
tensorrt_bin = r"D:\watermarkz\TensorRT-10.13.3.9\bin"
if tensorrt_bin not in os.environ["PATH"]:
    os.environ["PATH"] = tensorrt_bin + os.pathsep + os.environ["PATH"]

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
# NOTE: pycuda.autoinit removed - context initialized by sam2_trt_predictor.py
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAM2MemoryEncoderTRT:
    """
    TensorRT wrapper for SAM2 memory encoder

    The memory encoder takes image features and predicted masks,
    and generates memory features for temporal tracking.

    Inputs:
        - pix_feat: [B, 256, 64, 64] - Image features from encoder
        - masks: [B, 1, 1024, 1024] - Predicted masks

    Outputs:
        - vision_features: [B, 64, 64, 64] - Memory features
        - vision_pos_enc: [B, 64, 64, 64] - Positional encodings
    """

    def __init__(self, engine_path):
        """
        Initialize TensorRT memory encoder

        Args:
            engine_path: Path to TensorRT engine file
        """
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        logger.info(f"[MEMORY_ENCODER_TRT] Loading engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output tensor info
        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)

            tensor_info = {
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'mode': mode
            }

            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(tensor_info)
            else:
                self.outputs.append(tensor_info)

        logger.info(f"[MEMORY_ENCODER_TRT] Inputs: {[t['name'] for t in self.inputs]}")
        logger.info(f"[MEMORY_ENCODER_TRT] Outputs: {[t['name'] for t in self.outputs]}")
        logger.info(f"[MEMORY_ENCODER_TRT] Engine loaded successfully!")

        # Create CUDA stream once (reuse for all inferences to avoid memory leak)
        self.stream = cuda.Stream()

    def encode(self, pix_feat, masks):
        """
        Encode image features and masks into memory features

        Args:
            pix_feat: numpy array [B, 256, 64, 64] - Image features
            masks: numpy array [B, 1, 1024, 1024] - Predicted masks

        Returns:
            vision_features: numpy array [B, 64, 64, 64] - Memory features
            vision_pos_enc: numpy array [B, 64, 64, 64] - Positional encodings
        """
        # Ensure inputs are numpy arrays
        if not isinstance(pix_feat, np.ndarray):
            pix_feat = pix_feat.cpu().numpy() if hasattr(pix_feat, 'cpu') else np.array(pix_feat)
        if not isinstance(masks, np.ndarray):
            masks = masks.cpu().numpy() if hasattr(masks, 'cpu') else np.array(masks)

        # Ensure float32
        pix_feat = np.ascontiguousarray(pix_feat, dtype=np.float32)
        masks = np.ascontiguousarray(masks, dtype=np.float32)

        # Allocate GPU memory for inputs
        inputs_gpu = {}

        # Input 1: pix_feat
        d_pix_feat = cuda.mem_alloc(pix_feat.nbytes)
        cuda.memcpy_htod(d_pix_feat, pix_feat)
        inputs_gpu['pix_feat'] = d_pix_feat

        # Input 2: masks
        d_masks = cuda.mem_alloc(masks.nbytes)
        cuda.memcpy_htod(d_masks, masks)
        inputs_gpu['masks'] = d_masks

        # Allocate GPU memory for outputs
        outputs_gpu = {}
        outputs_host = {}

        for output_tensor in self.outputs:
            name = output_tensor['name']
            shape = output_tensor['shape']
            size = int(np.prod(shape)) * np.dtype(np.float32).itemsize

            outputs_gpu[name] = cuda.mem_alloc(size)
            outputs_host[name] = np.empty(shape, dtype=np.float32)

        # Bind inputs and outputs
        for input_tensor in self.inputs:
            name = input_tensor['name']
            if name in inputs_gpu:
                self.context.set_tensor_address(name, int(inputs_gpu[name]))

        for name, d_output in outputs_gpu.items():
            self.context.set_tensor_address(name, int(d_output))

        # Run inference with reused stream
        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

        # Copy outputs back to host
        for name, d_output in outputs_gpu.items():
            cuda.memcpy_dtoh(outputs_host[name], d_output)

        # Extract outputs
        vision_features = outputs_host.get('vision_features')
        vision_pos_enc = outputs_host.get('vision_pos_enc')

        if vision_features is None or vision_pos_enc is None:
            raise RuntimeError(f"Missing outputs. Available: {list(outputs_host.keys())}")

        # Synchronize both PyCUDA and PyTorch CUDA before freeing memory
        self.stream.synchronize()  # PyCUDA stream
        torch.cuda.synchronize()   # PyTorch CUDA context

        # Free GPU memory to prevent memory leak
        for d_buf in inputs_gpu.values():
            d_buf.free()
        for d_buf in outputs_gpu.values():
            d_buf.free()

        return vision_features, vision_pos_enc


def test_memory_encoder_trt():
    """Test the TensorRT memory encoder"""
    import time
    import pycuda.autoinit  # Needed for standalone test

    # Path to TensorRT engine
    engine_path = r"sam2_trt_inference/engines/sam2_memory_encoder_fp16.engine"

    # Initialize encoder
    print("="*70)
    print("TESTING SAM2 MEMORY ENCODER TENSORRT")
    print("="*70)

    encoder = SAM2MemoryEncoderTRT(engine_path)

    # Create dummy inputs
    print("\n[TEST] Creating dummy inputs...")
    pix_feat = np.random.randn(1, 256, 64, 64).astype(np.float32)
    masks = np.random.randn(1, 1, 1024, 1024).astype(np.float32)

    print(f"[INPUT] pix_feat: {pix_feat.shape}")
    print(f"[INPUT] masks: {masks.shape}")

    # Run inference
    print("\n[TEST] Running TensorRT inference...")
    start = time.time()
    vision_features, vision_pos_enc = encoder.encode(pix_feat, masks)
    elapsed = time.time() - start

    print(f"[OUTPUT] vision_features: {vision_features.shape}")
    print(f"[OUTPUT] vision_pos_enc: {vision_pos_enc.shape}")
    print(f"[TIME] Inference: {elapsed*1000:.2f}ms")

    # Benchmark
    print("\n[BENCHMARK] Running 100 iterations...")
    times = []
    for _ in range(100):
        start = time.time()
        vision_features, vision_pos_enc = encoder.encode(pix_feat, masks)
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000

    print(f"[RESULT] Average: {avg_time:.2f}ms")
    print(f"[RESULT] Std Dev: {std_time:.2f}ms")
    print(f"[RESULT] Min: {np.min(times)*1000:.2f}ms")
    print(f"[RESULT] Max: {np.max(times)*1000:.2f}ms")

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    test_memory_encoder_trt()
