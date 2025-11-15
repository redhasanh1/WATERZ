"""
SAM2 TensorRT Python Implementation
Fast, stable, Windows-compatible predictor for RTX 4090
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
import pycuda.autoinit
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAM2TensorRTPredictor:
    """Pure Python TensorRT SAM2 predictor - no C++ needed"""

    def __init__(self, encoder_engine_path, decoder_engine_path):
        self.encoder_engine_path = Path(encoder_engine_path)
        self.decoder_engine_path = Path(decoder_engine_path)

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load engines
        self.encoder_engine = self._load_engine(self.encoder_engine_path)
        self.decoder_engine = self._load_engine(self.decoder_engine_path)

        # Create execution contexts
        self.encoder_context = self.encoder_engine.create_execution_context()
        self.decoder_context = self.decoder_engine.create_execution_context()

        # Get tensor names (TensorRT 10+ API)
        self.encoder_input_name = self.encoder_engine.get_tensor_name(0)
        self.encoder_output_name = self.encoder_engine.get_tensor_name(1)

        # Image embeddings cache
        self.image_embeddings = None
        self.current_image_hash = None

        logger.info(f"[OK] Loaded encoder: {self.encoder_engine_path.name}")
        logger.info(f"[OK] Loaded decoder: {self.decoder_engine_path.name}")

    def _load_engine(self, engine_path):
        """Load TensorRT engine from file"""
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def set_image(self, image):
        """
        Encode image and cache embeddings

        Args:
            image: numpy array (H, W, 3) RGB uint8
        """
        # Check if we already encoded this image
        image_hash = hash(image.tobytes())
        if image_hash == self.current_image_hash:
            return  # Already encoded

        # Preprocess image to 1024x1024
        img_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        # Normalize to [-1, 1] and transpose to CHW
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1.0
        img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_chw, axis=0)  # Add batch dim -> (1, 3, 1024, 1024)
        img_batch = np.ascontiguousarray(img_batch)  # Ensure contiguous array for CUDA

        # Allocate GPU memory
        d_input = cuda.mem_alloc(img_batch.nbytes)

        # Get output shape from engine
        output_shape = self.encoder_context.get_tensor_shape(self.encoder_output_name)
        output_size = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
        d_output = cuda.mem_alloc(output_size)

        # Copy input to GPU
        cuda.memcpy_htod(d_input, img_batch)

        # Set tensor addresses
        self.encoder_context.set_tensor_address(self.encoder_input_name, int(d_input))
        self.encoder_context.set_tensor_address(self.encoder_output_name, int(d_output))

        # Run inference
        self.encoder_context.execute_async_v3(cuda.Stream().handle)

        # Copy output back to host
        embeddings = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(embeddings, d_output)

        # Cache embeddings
        self.image_embeddings = embeddings
        self.current_image_hash = image_hash

        logger.info(f"[OK] Encoded image: {embeddings.shape}")

    def predict(self, point_coords, point_labels):
        """
        Predict mask from point prompts

        Args:
            point_coords: numpy array of shape (N, 2) with (x, y) coordinates
            point_labels: numpy array of shape (N,) with 1 for positive, 0 for negative

        Returns:
            masks: numpy array (H, W) with predicted mask
            scores: confidence scores
        """
        if self.image_embeddings is None:
            raise ValueError("Call set_image() first to encode an image")

        # Prepare point prompts (scale to 1024x1024 space)
        points = np.array(point_coords, dtype=np.float32)
        labels = np.array(point_labels, dtype=np.float32)

        # Allocate decoder inputs/outputs
        # Input: image_embeddings, point_coords, point_labels
        # Output: masks, iou_predictions

        # Get decoder tensor names
        decoder_input_names = []
        decoder_output_names = []
        for i in range(self.decoder_engine.num_io_tensors):
            name = self.decoder_engine.get_tensor_name(i)
            if self.decoder_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                decoder_input_names.append(name)
            else:
                decoder_output_names.append(name)

        # Allocate GPU memory for decoder
        d_embeddings = cuda.mem_alloc(self.image_embeddings.nbytes)
        cuda.memcpy_htod(d_embeddings, self.image_embeddings)

        # Prepare point coordinates and labels
        point_coords_batch = np.expand_dims(points, axis=0)  # (1, N, 2)
        point_labels_batch = np.expand_dims(labels, axis=0)  # (1, N)

        d_points = cuda.mem_alloc(point_coords_batch.nbytes)
        d_labels = cuda.mem_alloc(point_labels_batch.nbytes)

        cuda.memcpy_htod(d_points, point_coords_batch)
        cuda.memcpy_htod(d_labels, point_labels_batch)

        # Get output shapes
        mask_shape = self.decoder_context.get_tensor_shape(decoder_output_names[0])
        iou_shape = self.decoder_context.get_tensor_shape(decoder_output_names[1])

        mask_size = int(np.prod(mask_shape)) * np.dtype(np.float32).itemsize
        iou_size = int(np.prod(iou_shape)) * np.dtype(np.float32).itemsize

        d_masks = cuda.mem_alloc(mask_size)
        d_iou = cuda.mem_alloc(iou_size)

        # Set tensor addresses
        self.decoder_context.set_tensor_address(decoder_input_names[0], int(d_embeddings))
        self.decoder_context.set_tensor_address(decoder_input_names[1], int(d_points))
        self.decoder_context.set_tensor_address(decoder_input_names[2], int(d_labels))
        self.decoder_context.set_tensor_address(decoder_output_names[0], int(d_masks))
        self.decoder_context.set_tensor_address(decoder_output_names[1], int(d_iou))

        # Run decoder
        self.decoder_context.execute_async_v3(cuda.Stream().handle)

        # Copy outputs
        masks_output = np.empty(mask_shape, dtype=np.float32)
        iou_output = np.empty(iou_shape, dtype=np.float32)

        cuda.memcpy_dtoh(masks_output, d_masks)
        cuda.memcpy_dtoh(iou_output, d_iou)

        # Post-process mask (sigmoid + threshold)
        mask_logits = masks_output[0, 0]  # Get first mask
        mask_probs = 1 / (1 + np.exp(-mask_logits))  # Sigmoid
        mask_binary = (mask_probs > 0.5).astype(np.uint8)

        logger.info(f"[OK] Predicted mask: {mask_binary.shape}, IoU: {iou_output[0, 0]:.3f}")

        return mask_binary, iou_output[0, 0]


def test_sam2_trt():
    """Test the TensorRT SAM2 predictor"""
    import time

    # Paths to TensorRT engines
    encoder_engine = r"D:\watermarkz\sam2_trt_inference\engines\sam2_encoder_fp16.engine"
    decoder_engine = r"D:\watermarkz\sam2_trt_inference\engines\sam2_decoder_fp16.engine"

    # Initialize predictor
    print("[*] Initializing SAM2 TensorRT predictor...")
    predictor = SAM2TensorRTPredictor(encoder_engine, decoder_engine)

    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Benchmark encoding
    print("\n[*] Benchmarking encoder...")
    start = time.perf_counter()
    predictor.set_image(test_image)
    encode_time = (time.perf_counter() - start) * 1000
    print(f"   Encoding time: {encode_time:.2f}ms")

    # Benchmark decoding
    print("\n[*] Benchmarking decoder...")
    point_coords = np.array([[256, 256]])  # Center point
    point_labels = np.array([1])  # Positive point

    times = []
    for i in range(10):
        start = time.perf_counter()
        mask, score = predictor.predict(point_coords, point_labels)
        decode_time = (time.perf_counter() - start) * 1000
        times.append(decode_time)
        if i == 0:
            print(f"   First run: {decode_time:.2f}ms")

    avg_time = np.mean(times[1:])  # Skip first run (warmup)
    print(f"   Average decode time: {avg_time:.2f}ms")
    print(f"   Total pipeline: {encode_time + avg_time:.2f}ms")

    print(f"\n[OK] SAM2 TensorRT working! Mask shape: {mask.shape}")
    print(f"[*] Target: <20ms total latency on RTX 4090")


if __name__ == "__main__":
    test_sam2_trt()
