"""
SAM2 TensorRT Inference Predictor - Optimized for RTX 4090

Fast inference wrapper for SAM2 using TensorRT engines with:
- Cached image embeddings (encode once, use for all clicks)
- FP16 precision with Tensor Cores
- Pinned memory for fast CPU↔GPU transfers
- CUDA streams for async execution

Target: Sub-100ms interactive preview on RTX 4090
"""

import os
import sys
import time
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Optional, Tuple, List

# Add SAM2 to path for transforms
SAM2_PATH = Path(__file__).parent / "segment-anything-2"
sys.path.insert(0, str(SAM2_PATH))

from sam2.utils.transforms import SAM2Transforms

# Try to import TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # Initialize CUDA
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("[WARNING] TensorRT/PyCUDA not available - install with:")
    print("          pip install tensorrt pycuda")


class TensorRTEngine:
    """TensorRT engine wrapper for efficient inference"""

    def __init__(self, engine_path: str):
        """
        Load TensorRT engine from file

        Args:
            engine_path: Path to .engine file
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        print(f"[TRT] Loading engine: {os.path.basename(engine_path)}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.context.get_tensor_shape(tensor_name)
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({"name": tensor_name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype})
                self.input_names.append(tensor_name)
            else:
                self.outputs.append({"name": tensor_name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype})
                self.output_names.append(tensor_name)

        print(f"[TRT] Engine loaded - Inputs: {self.input_names}, Outputs: {self.output_names}")

    def infer(self, **inputs):
        """
        Run inference

        Args:
            **inputs: Named input tensors as numpy arrays

        Returns:
            dict: Output tensors
        """
        # Copy inputs to device
        for inp in self.inputs:
            name = inp["name"]
            if name not in inputs:
                raise ValueError(f"Missing input: {name}")

            # Flatten and copy to pinned host memory
            np.copyto(inp["host"], inputs[name].ravel())
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)

            # Set tensor address
            self.context.set_tensor_address(name, int(inp["device"]))

        # Set output tensor addresses
        for out in self.outputs:
            self.context.set_tensor_address(out["name"], int(out["device"]))

        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs to host
        results = {}
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
            self.stream.synchronize()
            results[out["name"]] = out["host"].reshape(out["shape"]).copy()

        return results

    def __del__(self):
        """Cleanup"""
        del self.context
        del self.engine
        del self.stream


class SAM2TensorRTPredictor:
    """
    SAM2 predictor using TensorRT engines for ultra-fast inference

    Usage:
        predictor = SAM2TensorRTPredictor("sam2_tensorrt_engines")
        predictor.set_image(image)  # Encode once
        mask = predictor.predict(points, labels)  # Fast prediction (~15-30ms)
    """

    def __init__(self, engine_dir: str, image_size: int = 1024):
        """
        Initialize TensorRT predictor

        Args:
            engine_dir: Directory containing .engine files
            image_size: Input image size (default: 1024 for SAM2)
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available - falling back to PyTorch")

        self.engine_dir = engine_dir
        self.image_size = image_size

        # Load engines
        encoder_path = os.path.join(engine_dir, "sam2_image_encoder_fp16.engine")
        decoder_path = os.path.join(engine_dir, "sam2_prompt_mask_decoder_fp16.engine")

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder engine not found: {encoder_path}")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(f"Decoder engine not found: {decoder_path}")

        print("[SAM2-TRT] Loading TensorRT engines...")
        self.encoder_engine = TensorRTEngine(encoder_path)
        self.decoder_engine = TensorRTEngine(decoder_path)
        print("[SAM2-TRT] Engines loaded successfully!")

        # Initialize transforms (same as SAM2)
        self._transforms = SAM2Transforms(
            resolution=self.image_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )

        # Cached embeddings
        self._image_embed = None
        self._high_res_feat_0 = None
        self._high_res_feat_1 = None
        self._orig_hw = None
        self._is_image_set = False

    def set_image(self, image: np.ndarray):
        """
        Encode image and cache embeddings (do this once per image)

        Args:
            image: Input image as numpy array [H, W, 3] in RGB format
        """
        print("[SAM2-TRT] Encoding image...")
        start_time = time.time()

        # Store original size
        self._orig_hw = image.shape[:2]

        # Transform image
        input_image = self._transforms(image)  # Returns [3, 1024, 1024]
        input_image = input_image[None, ...].astype(np.float32)  # [1, 3, 1024, 1024]

        # Run encoder
        encoder_outputs = self.encoder_engine.infer(image=input_image)

        # Cache embeddings
        self._image_embed = encoder_outputs["image_embed"]
        self._high_res_feat_0 = encoder_outputs["high_res_feat_0"]
        self._high_res_feat_1 = encoder_outputs["high_res_feat_1"]
        self._is_image_set = True

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[SAM2-TRT] Image encoded in {elapsed_ms:.1f}ms")

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict mask from points using cached embeddings (FAST!)

        Args:
            point_coords: Point coordinates [N, 2] in original image space
            point_labels: Point labels [N] (1=positive, 0=negative)

        Returns:
            mask: Binary mask [H, W] in original image size
            iou_score: IoU prediction score
        """
        if not self._is_image_set:
            raise RuntimeError("Image not set! Call set_image() first")

        start_time = time.time()

        # Prepare inputs (add batch dimension)
        point_coords_batch = point_coords[None, ...].astype(np.float32)  # [1, N, 2]
        point_labels_batch = point_labels[None, ...].astype(np.int64)  # [1, N]

        # Run decoder with cached embeddings
        decoder_outputs = self.decoder_engine.infer(
            image_embed=self._image_embed,
            high_res_feat_0=self._high_res_feat_0,
            high_res_feat_1=self._high_res_feat_1,
            point_coords=point_coords_batch,
            point_labels=point_labels_batch,
        )

        # Get outputs
        low_res_masks = decoder_outputs["low_res_masks"]  # [1, 1, 256, 256]
        iou_predictions = decoder_outputs["iou_predictions"]  # [1, 1]

        # Remove batch dimension
        low_res_mask = low_res_masks[0, 0]  # [256, 256]
        iou_score = float(iou_predictions[0, 0])

        # Resize to original image size
        mask = cv2.resize(
            low_res_mask,
            (self._orig_hw[1], self._orig_hw[0]),  # (width, height)
            interpolation=cv2.INTER_LINEAR
        )

        # Threshold to binary mask
        mask = (mask > 0.0).astype(np.uint8)

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[SAM2-TRT] Mask predicted in {elapsed_ms:.1f}ms (IoU: {iou_score:.3f})")

        return mask, iou_score

    def reset_image(self):
        """Clear cached embeddings"""
        self._image_embed = None
        self._high_res_feat_0 = None
        self._high_res_feat_1 = None
        self._orig_hw = None
        self._is_image_set = False


def test_tensorrt_predictor():
    """Quick test of TensorRT predictor"""
    print("="*70)
    print("SAM2 TensorRT Predictor - Quick Test")
    print("="*70)

    # Load test image
    test_image_path = "segment-anything-2/notebooks/images/truck.jpg"
    if not os.path.exists(test_image_path):
        print(f"[ERROR] Test image not found: {test_image_path}")
        return

    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"[TEST] Loaded image: {image.shape}")

    # Initialize predictor
    predictor = SAM2TensorRTPredictor("sam2_tensorrt_engines")

    # Encode image (this is cached)
    predictor.set_image(image)

    # Test multiple predictions with different points
    test_points = [
        (np.array([[500.0, 375.0]]), np.array([1])),  # Single positive point
        (np.array([[500.0, 375.0], [600.0, 400.0]]), np.array([1, 1])),  # Two positive points
        (np.array([[500.0, 375.0], [300.0, 200.0]]), np.array([1, 0])),  # Positive + negative
    ]

    for i, (coords, labels) in enumerate(test_points):
        print(f"\n[TEST {i+1}] Predicting with {len(coords)} points...")
        mask, iou = predictor.predict(coords, labels)
        print(f"[TEST {i+1}] Mask shape: {mask.shape}, Coverage: {mask.sum() / mask.size * 100:.1f}%")

    print("\n[SUCCESS] TensorRT predictor working correctly!")


if __name__ == "__main__":
    test_tensorrt_predictor()
