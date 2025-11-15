"""
SAM2 TensorRT Fast Predictor
Optimized wrapper for TIER IV's TensorRT implementation with point-based prompts
Target: 11-17ms interactive mask prediction on RTX 4090
"""

import ctypes
import numpy as np
from typing import List, Tuple, Optional
import os


class SAM2TRTFastPredictor:
    """
    Ultra-fast SAM2 predictor using TensorRT engines (C++ backend)

    Performance targets on RTX 4090:
    - set_image(): 8-12ms (one-time per frame)
    - predict(): 3-5ms (per click update)
    - Total interactive: 11-17ms (7-10x faster than PyTorch)
    """

    def __init__(self,
                 encoder_engine: str,
                 decoder_engine: str,
                 dll_path: str = None):
        """
        Initialize TensorRT SAM2 predictor

        Args:
            encoder_engine: Path to encoder TensorRT engine (.engine file)
            decoder_engine: Path to decoder TensorRT engine (.engine file)
            dll_path: Path to trt_sam2_infer.dll (auto-detect if None)
        """
        # Auto-detect DLL if not provided
        if dll_path is None:
            dll_path = self._find_dll()

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found: {dll_path}")

        if not os.path.exists(encoder_engine):
            raise FileNotFoundError(f"Encoder engine not found: {encoder_engine}")

        if not os.path.exists(decoder_engine):
            raise FileNotFoundError(f"Decoder engine not found: {decoder_engine}")

        # Load DLL
        self.lib = ctypes.CDLL(dll_path)

        # Initialize OpenCV threads
        self.lib.InitOpenCVThreads()

        # Create SAM2 instance
        self.lib.create_sam2image.argtypes = [
            ctypes.c_char_p,  # encoder_path
            ctypes.c_char_p,  # decoder_path
            ctypes.c_char_p,  # precision
            ctypes.c_int      # decoder_batch_limit
        ]
        self.lib.create_sam2image.restype = ctypes.c_void_p

        self.instance = self.lib.create_sam2image(
            encoder_engine.encode('utf-8'),
            decoder_engine.encode('utf-8'),
            b"fp16",
            10  # Support up to 10 points
        )

        if not self.instance:
            raise RuntimeError("Failed to create SAM2 TensorRT instance")

        # Set up function signatures
        self._setup_function_signatures()

        # Cache image dimensions
        self.orig_size = None

    def _find_dll(self) -> str:
        """Auto-detect trt_sam2_infer.dll location"""
        search_paths = [
            "D:/watermarkz/sam2_trt_inference/build/Release/trt_sam2_infer.dll",
            "D:/watermarkz/sam2_trt_inference/build/Debug/trt_sam2_infer.dll",
            "./sam2_trt_inference/build/Release/trt_sam2_infer.dll",
            "./trt_sam2_infer.dll",
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "Could not find trt_sam2_infer.dll. Please build the C++ library first."
        )

    def _setup_function_signatures(self):
        """Set up ctypes function signatures for C bindings"""

        # sam2image_set_image
        self.lib.sam2image_set_image.argtypes = [
            ctypes.c_void_p,                          # instance
            ctypes.POINTER(ctypes.c_ubyte),           # image_data
            ctypes.c_int,                              # width
            ctypes.c_int                               # height
        ]
        self.lib.sam2image_set_image.restype = None

        # sam2image_predict_points
        self.lib.sam2image_predict_points.argtypes = [
            ctypes.c_void_p,                          # instance
            ctypes.POINTER(ctypes.c_float),           # point_coords
            ctypes.POINTER(ctypes.c_float),           # point_labels
            ctypes.c_int,                              # num_points
            ctypes.POINTER(ctypes.c_ubyte),           # output_mask
            ctypes.c_int,                              # width
            ctypes.c_int                               # height
        ]
        self.lib.sam2image_predict_points.restype = None

        # destroy_sam2image
        self.lib.destroy_sam2image.argtypes = [ctypes.c_void_p]
        self.lib.destroy_sam2image.restype = None

    def set_image(self, image: np.ndarray):
        """
        Encode image (do once, cache embeddings)

        Args:
            image: RGB image as uint8 numpy array [H, W, 3]

        Performance: ~8-12ms on RTX 4090
        """
        if image.dtype != np.uint8:
            raise ValueError("Image must be uint8")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB [H, W, 3]")

        height, width, _ = image.shape
        self.orig_size = (height, width)

        # Convert to contiguous array if needed
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        # Call C++ encoder
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.lib.sam2image_set_image(self.instance, img_data, width, height)

    def predict(self,
                point_coords: np.ndarray,
                point_labels: np.ndarray,
                multimask_output: bool = False) -> np.ndarray:
        """
        Predict mask from point clicks (uses cached image embeddings)

        Args:
            point_coords: Point coordinates [N, 2] in original image space (x, y)
            point_labels: Point labels [N] (1=foreground, 0=background)
            multimask_output: Ignored (kept for compatibility)

        Returns:
            mask: Binary mask [H, W] as uint8

        Performance: ~3-5ms on RTX 4090
        """
        if self.orig_size is None:
            raise RuntimeError("Must call set_image() first")

        if len(point_coords) == 0:
            raise ValueError("Must provide at least one point")

        if len(point_coords) != len(point_labels):
            raise ValueError("point_coords and point_labels must have same length")

        # Convert to float32 arrays
        coords_flat = np.array(point_coords, dtype=np.float32).flatten()
        labels_flat = np.array(point_labels, dtype=np.float32).flatten()

        # Prepare output mask
        height, width = self.orig_size
        output_mask = np.zeros((height, width), dtype=np.uint8)

        # Call C++ decoder
        coords_ptr = coords_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        labels_ptr = labels_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mask_ptr = output_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        self.lib.sam2image_predict_points(
            self.instance,
            coords_ptr,
            labels_ptr,
            len(point_coords),
            mask_ptr,
            width,
            height
        )

        return output_mask

    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'instance') and self.instance:
            self.lib.destroy_sam2image(self.instance)


def test_predictor():
    """Quick test of the TensorRT predictor"""
    import time

    # Paths
    encoder_engine = "D:/watermarkz/sam2_trt_inference/engines/sam2_encoder_fp16.engine"
    decoder_engine = "D:/watermarkz/sam2_trt_inference/engines/sam2_decoder_fp16.engine"

    print("Initializing SAM2 TensorRT predictor...")
    predictor = SAM2TRTFastPredictor(encoder_engine, decoder_engine)
    print("Initialized!")

    # Create test image
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # Test encoding
    print("\nTesting image encoding...")
    start = time.perf_counter()
    predictor.set_image(test_image)
    encode_time = (time.perf_counter() - start) * 1000
    print(f"Encoding time: {encode_time:.1f}ms")

    # Test point prediction
    print("\nTesting point prediction...")
    point_coords = np.array([[960, 540], [800, 400]])  # Two points
    point_labels = np.array([1, 1])  # Both positive

    start = time.perf_counter()
    mask = predictor.predict(point_coords, point_labels)
    predict_time = (time.perf_counter() - start) * 1000
    print(f"Prediction time: {predict_time:.1f}ms")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")

    # Total interactive time
    total_time = encode_time + predict_time
    print(f"\nTotal interactive time: {total_time:.1f}ms")

    if total_time < 20:
        print("SUCCESS! Target achieved (< 20ms)")
    else:
        print(f"Not quite there yet (target: < 20ms, got: {total_time:.1f}ms)")


if __name__ == "__main__":
    test_predictor()
