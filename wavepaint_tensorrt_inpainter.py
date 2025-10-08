import os
import sys
from typing import Optional

sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import torch


class WavePaintTensorRTInpainter:
    """
    TensorRT-backed WavePaint with full-frame preprocessing to match
    the behaviour of test_wavepaint_only.py (global resize → inpaint → upscale).
    Adds gentle temporal smoothing inside the mask to avoid flicker.
    """

    def __init__(self, engine_path: str = 'weights/wavepaint.engine'):
        self.engine_path = engine_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.prev_result: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None

        if self.device == 'cpu':
            raise RuntimeError("WavePaint TensorRT requires CUDA (GPU)")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        try:
            import tensorrt as trt

            print(f"Loading WavePaint TensorRT engine from {engine_path}...")
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            with open(engine_path, 'rb') as f:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)

            self.context = self.engine.create_execution_context()

            self.d_input_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32).cuda()
            self.d_input_mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32).cuda()
            self.d_output = torch.zeros((1, 3, 256, 256), dtype=torch.float32).cuda()

            self.bindings = [
                int(self.d_input_img.data_ptr()),
                int(self.d_input_mask.data_ptr()),
                int(self.d_output.data_ptr()),
            ]

            print("✅ WavePaint TensorRT engine loaded successfully!")
        except ImportError as e:
            raise RuntimeError("TensorRT Python bindings not available") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}") from e

    def _prepare_inputs(self, image: np.ndarray, mask: np.ndarray):
        img_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)

        img_float = img_resized.astype(np.float32) / 255.0
        mask_float = (mask_resized.astype(np.float32) / 255.0)[..., np.newaxis]

        masked_img = img_float * (1 - mask_float)

        img_input = masked_img.transpose(2, 0, 1)
        mask_input = mask_resized[np.newaxis, :, :] / 255.0

        return img_input, mask_input

    def _run_tensorrt(self, img_input: np.ndarray, mask_input: np.ndarray) -> np.ndarray:
        self.d_input_img.copy_(torch.from_numpy(img_input).unsqueeze(0))
        self.d_input_mask.copy_(torch.from_numpy(mask_input).unsqueeze(0))
        self.context.execute_v2(self.bindings)
        result = self.d_output.cpu().numpy()[0]
        result = result.transpose(1, 2, 0)
        result = (result * 255).clip(0, 255).astype(np.uint8)
        return result

    def inpaint_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image is None or mask is None:
            raise ValueError("image and mask must be numpy arrays")

        if mask.max() < 127:
            # Nothing to inpaint; reset temporal cache
            self.prev_result = None
            self.prev_mask = None
            return image

        h, w = image.shape[:2]

        img_input, mask_input = self._prepare_inputs(image, mask)
        output_256 = self._run_tensorrt(img_input, mask_input)

        output_full = cv2.resize(output_256, (w, h), interpolation=cv2.INTER_CUBIC)

        mask_dilated = cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=1)
        mask_smooth = cv2.GaussianBlur(mask_dilated.astype(np.float32), (61, 61), 0) / 255.0
        mask_smooth = np.clip(mask_smooth, 0.0, 1.0)

        if mask_smooth.ndim == 2:
            mask_smooth = np.stack([mask_smooth] * 3, axis=2)

        blended = (
            output_full.astype(np.float32) * mask_smooth
            + image.astype(np.float32) * (1 - mask_smooth)
        )
        result_frame = blended.astype(np.uint8)

        if (
            self.prev_result is not None
            and self.prev_result.shape == result_frame.shape
            and self.prev_mask is not None
        ):
            temporal_mask = np.maximum(mask_smooth, self.prev_mask)
            temporal_strength = 0.25  # closer to 0 = keep more detail
            result_frame = (
                result_frame.astype(np.float32) * (1 - temporal_mask * temporal_strength)
                + self.prev_result.astype(np.float32) * (temporal_mask * temporal_strength)
            ).astype(np.uint8)

        self.prev_result = result_frame.copy()
        self.prev_mask = mask_smooth

        return result_frame
