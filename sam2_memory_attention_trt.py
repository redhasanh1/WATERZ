"""
SAM2 Memory Attention - TensorRT Wrapper

Provides a TensorRT-based implementation of the SAM2 memory attention module.
This replaces the PyTorch memory attention for ultra-fast video tracking.

This is the FINAL PIECE for full TensorRT SAM2 video tracking!
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


class SAM2MemoryAttentionTRT:
    """
    TensorRT wrapper for SAM2 memory attention

    The memory attention performs cross-attention between current frame features
    and memory bank features for temporal tracking.

    Inputs:
        - curr: [HW, B, C] - Current frame vision features
        - curr_pos: [HW, B, C] - Current frame positional encoding
        - memory: [M, B, mem_dim] - Memory bank features (variable length M)
        - memory_pos: [M, B, mem_dim] - Memory positional encodings
        - num_obj_ptr_tokens: int - Number of object pointer tokens (usually 0)

    Outputs:
        - pix_feat_with_mem: [HW, B, C] - Memory-fused features
    """

    def __init__(self, engine_path):
        """
        Initialize TensorRT memory attention

        Args:
            engine_path: Path to TensorRT engine file
        """
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        logger.info(f"[MEMORY_ATTENTION_TRT] Loading engine: {engine_path}")
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

        logger.info(f"[MEMORY_ATTENTION_TRT] Inputs: {[t['name'] for t in self.inputs]}")
        logger.info(f"[MEMORY_ATTENTION_TRT] Outputs: {[t['name'] for t in self.outputs]}")
        logger.info(f"[MEMORY_ATTENTION_TRT] Engine loaded successfully!")

        # Create CUDA stream once (reuse for all inferences to avoid memory leak)
        self.stream = cuda.Stream()

    def forward(self, curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens=0):
        """
        Perform memory attention inference

        Args:
            curr: numpy array [HW, B, C] - Current frame features
            curr_pos: numpy array [HW, B, C] - Current frame positional encoding
            memory: numpy array [M, B, mem_dim] - Memory bank features
            memory_pos: numpy array [M, B, mem_dim] - Memory positional encodings
            num_obj_ptr_tokens: int - Number of object pointer tokens (default 0)

        Returns:
            pix_feat_with_mem: numpy array [HW, B, C] - Memory-fused features
        """
        # Ensure inputs are numpy arrays
        if not isinstance(curr, np.ndarray):
            curr = curr.cpu().numpy() if hasattr(curr, 'cpu') else np.array(curr)
        if not isinstance(curr_pos, np.ndarray):
            curr_pos = curr_pos.cpu().numpy() if hasattr(curr_pos, 'cpu') else np.array(curr_pos)
        if not isinstance(memory, np.ndarray):
            memory = memory.cpu().numpy() if hasattr(memory, 'cpu') else np.array(memory)
        if not isinstance(memory_pos, np.ndarray):
            memory_pos = memory_pos.cpu().numpy() if hasattr(memory_pos, 'cpu') else np.array(memory_pos)

        # Ensure float32 and contiguous
        curr = np.ascontiguousarray(curr, dtype=np.float32)
        curr_pos = np.ascontiguousarray(curr_pos, dtype=np.float32)
        memory = np.ascontiguousarray(memory, dtype=np.float32)
        memory_pos = np.ascontiguousarray(memory_pos, dtype=np.float32)

        # Set dynamic shapes
        self.context.set_input_shape('curr', curr.shape)
        self.context.set_input_shape('curr_pos', curr_pos.shape)
        self.context.set_input_shape('memory', memory.shape)
        self.context.set_input_shape('memory_pos', memory_pos.shape)

        # Allocate GPU memory for inputs
        inputs_gpu = {}

        d_curr = cuda.mem_alloc(curr.nbytes)
        cuda.memcpy_htod(d_curr, curr)
        inputs_gpu['curr'] = d_curr

        d_curr_pos = cuda.mem_alloc(curr_pos.nbytes)
        cuda.memcpy_htod(d_curr_pos, curr_pos)
        inputs_gpu['curr_pos'] = d_curr_pos

        d_memory = cuda.mem_alloc(memory.nbytes)
        cuda.memcpy_htod(d_memory, memory)
        inputs_gpu['memory'] = d_memory

        d_memory_pos = cuda.mem_alloc(memory_pos.nbytes)
        cuda.memcpy_htod(d_memory_pos, memory_pos)
        inputs_gpu['memory_pos'] = d_memory_pos

        # Allocate GPU memory for outputs
        outputs_gpu = {}
        outputs_host = {}

        for output_tensor in self.outputs:
            name = output_tensor['name']
            # Query actual shape after input shapes are set
            actual_shape = self.context.get_tensor_shape(name)
            size = int(np.prod(actual_shape)) * np.dtype(np.float32).itemsize

            outputs_gpu[name] = cuda.mem_alloc(size)
            outputs_host[name] = np.empty(actual_shape, dtype=np.float32)

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

        # Extract output
        pix_feat_with_mem = outputs_host.get('pix_feat_with_mem')

        if pix_feat_with_mem is None:
            raise RuntimeError(f"Missing output. Available: {list(outputs_host.keys())}")

        # Synchronize both PyCUDA and PyTorch CUDA before freeing memory
        self.stream.synchronize()
        torch.cuda.synchronize()

        # Free GPU memory to prevent memory leak
        for d_buf in inputs_gpu.values():
            d_buf.free()
        for d_buf in outputs_gpu.values():
            d_buf.free()

        return pix_feat_with_mem

    def __call__(self, curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens=0):
        """Allows calling instance like a function"""
        return self.forward(curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens)


def test_memory_attention_trt():
    """Test the TensorRT memory attention"""
    import time
    import pycuda.autoinit  # Needed for standalone test

    # Path to TensorRT engine
    engine_path = r"sam2_trt_inference/engines/sam2_memory_attention_fp16.engine"

    # Initialize memory attention
    print("="*70)
    print("TESTING SAM2 MEMORY ATTENTION TENSORRT")
    print("="*70)

    mem_attn = SAM2MemoryAttentionTRT(engine_path)

    # Create dummy inputs
    print("\n[TEST] Creating dummy inputs...")
    HW = 64 * 64
    B = 1
    C = 256
    M = 7 * 4096  # 7 frames of memory
    mem_dim = 64

    curr = np.random.randn(HW, B, C).astype(np.float32)
    curr_pos = np.random.randn(HW, B, C).astype(np.float32)
    memory = np.random.randn(M, B, mem_dim).astype(np.float32)
    memory_pos = np.random.randn(M, B, mem_dim).astype(np.float32)

    print(f"[INPUT] curr: {curr.shape}")
    print(f"[INPUT] curr_pos: {curr_pos.shape}")
    print(f"[INPUT] memory: {memory.shape}")
    print(f"[INPUT] memory_pos: {memory_pos.shape}")

    # Run inference
    print("\n[TEST] Running TensorRT inference...")
    start = time.time()
    output = mem_attn(curr, curr_pos, memory, memory_pos)
    elapsed = time.time() - start

    print(f"[OUTPUT] pix_feat_with_mem: {output.shape}")
    print(f"[TIME] Inference: {elapsed*1000:.2f}ms")

    # Benchmark
    print("\n[BENCHMARK] Running 100 iterations...")
    times = []
    for _ in range(100):
        start = time.time()
        output = mem_attn(curr, curr_pos, memory, memory_pos)
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
    test_memory_attention_trt()
