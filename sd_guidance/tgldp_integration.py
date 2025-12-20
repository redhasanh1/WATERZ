"""
TGLDP v2.0 Integration Module

Temporal-Guided Latent Diffusion Propagation for ProPainter.

This is the main orchestrator that:
1. Batch-analyzes video to find hard frames (15% of video)
2. Pre-generates SD latents for ONLY those frames
3. Injects latents deep into ProPainter's encoder

Unlike v1 (broken post-processing), v2 does DEEP injection.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)

# Environment variable controls
USE_SD_GUIDANCE = os.environ.get('USE_SD_GUIDANCE', '0') == '1'
SD_INFERENCE_STEPS = int(os.environ.get('SD_INFERENCE_STEPS', '3'))
SD_GUIDANCE_SCALE = float(os.environ.get('SD_GUIDANCE_SCALE', '7.5'))
SD_HARD_REGION_THRESHOLD = float(os.environ.get('SD_HARD_REGION_THRESHOLD', '0.15'))  # 15% of frames
FIRSTPHOTO_PATH = os.environ.get('BACKGROUND_IMAGE_PATH', 'D:\\watermarkz\\videostotrain\\firstphoto.jpg')


class TGLDPv2Module:
    """
    TGLDP v2.0: Deep SD Latent Injection into ProPainter

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. BatchDifficultyAnalyzer: Scan ALL frames, identify hard 15% │
    ├─────────────────────────────────────────────────────────────────┤
    │ 2. SDPreprocessor: Generate latents ONLY for hard frames       │
    │    - Uses firstphoto.jpg as prior (fast)                       │
    │    - 3-step denoising (not 50!)                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ 3. SDGuidedInpaintGenerator: Inject at encoder level           │
    │    - SD latent → Project to 128ch → Blend with encoder output  │
    │    - Temporal consistency via memory bank                      │
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        firstphoto_path: str = None,
        hard_frame_ratio: float = 0.15,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            firstphoto_path: Path to clean background reference
            hard_frame_ratio: Fraction of frames to enhance (0.15 = 15%)
            device: Device for computation
            dtype: Data type (fp16 for 4090)
        """
        self.device = device
        self.dtype = dtype
        self.firstphoto_path = firstphoto_path or FIRSTPHOTO_PATH
        self.hard_frame_ratio = hard_frame_ratio

        # Components (lazy-loaded)
        self._analyzer = None
        self._preprocessor = None
        self._sd_model = None

        # Stats
        self._last_hard_frames = []
        self._last_processing_time = 0

        self._initialized = False

    def _init_components(self):
        """Lazy-initialize components on first use."""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("TGLDP v2.0 INITIALIZATION")
        logger.info("=" * 60)

        # 1. Batch analyzer
        from .batch_analyzer import BatchDifficultyAnalyzer
        self._analyzer = BatchDifficultyAnalyzer(
            hard_frame_ratio=self.hard_frame_ratio,
            device=self.device,
        )
        logger.info("[1/3] BatchDifficultyAnalyzer ready")

        # 2. SD Preprocessor (with firstphoto)
        from .preprocessor import SDPreprocessor
        self._preprocessor = SDPreprocessor(
            firstphoto_path=self.firstphoto_path,
            device=self.device,
            dtype=self.dtype,
        )
        logger.info(f"[2/3] SDPreprocessor ready (firstphoto: {os.path.exists(self.firstphoto_path)})")

        # 3. SD model is loaded inside preprocessor when needed
        logger.info("[3/3] SD model will load on demand")

        self._initialized = True
        logger.info("=" * 60)

    def analyze_and_pregenerate(
        self,
        frames: torch.Tensor,       # [T, C, H, W]
        masks: torch.Tensor,        # [T, 1, H, W]
        flows: Optional[torch.Tensor] = None,  # [T-1, 2, H, W]
    ) -> Tuple[List[int], Dict[int, torch.Tensor]]:
        """
        Phase 1 & 2: Analyze video and pre-generate SD latents.

        This should be called BEFORE the main ProPainter loop.

        Args:
            frames: Video frames tensor
            masks: Binary masks
            flows: Optional optical flow

        Returns:
            hard_indices: List of frame indices needing SD
            sd_latents: {frame_idx: latent_tensor}
        """
        self._init_components()

        start_time = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("TGLDP v2.0: ANALYSIS & PRE-GENERATION")
        logger.info("=" * 60)

        # Ensure tensors are on device
        frames = frames.to(self.device).to(self.dtype)
        masks = masks.to(self.device).to(self.dtype)

        T = frames.shape[0]
        logger.info(f"Video: {T} frames @ {frames.shape[-2]}x{frames.shape[-1]}")

        # Step 1: Analyze difficulty
        hard_indices, difficulty_scores = self._analyzer.analyze_video(
            frames, masks, flows
        )
        self._last_hard_frames = hard_indices

        logger.info(f"Hard frames: {len(hard_indices)}/{T} ({len(hard_indices)/T*100:.1f}%)")
        logger.info(f"Indices: {hard_indices}")

        # Step 2: Pre-generate SD latents for hard frames only
        if len(hard_indices) > 0:
            sd_latents = self._preprocessor.generate_keyframe_latents(
                frames=frames,
                masks=masks,
                keyframe_indices=hard_indices,
                use_firstphoto_prior=os.path.exists(self.firstphoto_path),
            )
        else:
            sd_latents = {}
            logger.info("No hard frames - skipping SD generation")

        # Optional: Propagate to neighbors for smooth transitions
        if flows is not None and len(sd_latents) > 0:
            sd_latents = self._preprocessor.propagate_latents_temporally(
                sd_latents, T, flows
            )

        self._last_processing_time = time.time() - start_time
        logger.info(f"Pre-generation complete: {self._last_processing_time:.2f}s")
        logger.info("=" * 60 + "\n")

        return hard_indices, sd_latents

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'initialized': self._initialized,
            'hard_frames': self._last_hard_frames,
            'num_hard_frames': len(self._last_hard_frames),
            'processing_time': self._last_processing_time,
            'firstphoto_loaded': self._preprocessor is not None and
                               self._preprocessor.firstphoto_tensor is not None,
        }


# Global singleton
_tgldp_v2_instance: Optional[TGLDPv2Module] = None


def get_tgldp_v2(
    firstphoto_path: str = None,
    force_reinit: bool = False,
) -> Optional[TGLDPv2Module]:
    """
    Get or create TGLDP v2 singleton.

    Args:
        firstphoto_path: Path to clean background reference
        force_reinit: Force re-initialization

    Returns:
        TGLDPv2Module instance or None if disabled
    """
    global _tgldp_v2_instance

    if not USE_SD_GUIDANCE:
        logger.info("TGLDP v2 disabled (USE_SD_GUIDANCE=0)")
        return None

    if _tgldp_v2_instance is None or force_reinit:
        _tgldp_v2_instance = TGLDPv2Module(
            firstphoto_path=firstphoto_path or FIRSTPHOTO_PATH,
        )

    return _tgldp_v2_instance


def process_video_with_tgldp(
    frames: torch.Tensor,
    masks: torch.Tensor,
    flows: Optional[torch.Tensor] = None,
    propainter_model=None,
    propainter_kwargs: dict = None,
) -> torch.Tensor:
    """
    Full TGLDP v2 pipeline.

    This is the main entry point for processing a video with SD guidance.

    Args:
        frames: [T, C, H, W] video frames
        masks: [T, 1, H, W] binary masks
        flows: [T-1, 2, H, W] optical flow (optional)
        propainter_model: SDGuidedInpaintGenerator instance
        propainter_kwargs: Additional kwargs for ProPainter

    Returns:
        restored: [T, C, H, W] inpainted frames
    """
    tgldp = get_tgldp_v2()
    if tgldp is None:
        logger.warning("TGLDP disabled - using standard ProPainter")
        return None

    # Phase 1 & 2: Analyze and pre-generate
    hard_indices, sd_latents = tgldp.analyze_and_pregenerate(
        frames, masks, flows
    )

    if propainter_model is None:
        # Just return the latents for external use
        return hard_indices, sd_latents

    # Phase 3: Run SD-guided ProPainter
    propainter_kwargs = propainter_kwargs or {}

    with torch.inference_mode():
        restored = propainter_model(
            **propainter_kwargs,
            sd_latents=sd_latents,
            hard_frame_indices=hard_indices,
        )

    return restored


# ============================================================================
# Legacy compatibility - these are DEPRECATED, use TGLDPv2Module instead
# ============================================================================

class GradientDifficultyDetector:
    """DEPRECATED: Use BatchDifficultyAnalyzer instead."""

    def __init__(self, *args, **kwargs):
        logger.warning("GradientDifficultyDetector is deprecated. Use BatchDifficultyAnalyzer.")
        from .batch_analyzer import BatchDifficultyAnalyzer
        self._analyzer = BatchDifficultyAnalyzer(*args, **kwargs)

    def compute_difficulty_mask(self, *args, **kwargs):
        return self._analyzer.analyze_video(*args, **kwargs)


class TGLDPEnhancer:
    """DEPRECATED: Use TGLDPv2Module instead."""

    def __init__(self, *args, **kwargs):
        logger.warning("TGLDPEnhancer (v1) is deprecated. Use TGLDPv2Module.")
        self._v2 = TGLDPv2Module(*args, **kwargs)

    def enhance_batch(self, *args, **kwargs):
        logger.warning("enhance_batch is deprecated. Use analyze_and_pregenerate + SDGuidedInpaintGenerator")
        return kwargs.get('propainter_output', None)


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing TGLDP v2.0...")

    # Force enable for testing
    os.environ['USE_SD_GUIDANCE'] = '1'

    # Create module
    tgldp = TGLDPv2Module(
        firstphoto_path="D:\\watermarkz\\videostotrain\\firstphoto.jpg",
        hard_frame_ratio=0.15,
    )

    # Create dummy data
    T, C, H, W = 30, 3, 480, 640
    frames = torch.rand(T, C, H, W, device="cuda", dtype=torch.float16)
    masks = torch.zeros(T, 1, H, W, device="cuda", dtype=torch.float16)

    # Create varying masks
    for i in range(T):
        size = 100 + (i % 10) * 20
        y = H // 2 - size // 2
        x = W // 2 - size // 2
        masks[i, 0, y:y+size, x:x+size] = 1

    # Analyze and pre-generate
    hard_indices, sd_latents = tgldp.analyze_and_pregenerate(frames, masks)

    print(f"\nResults:")
    print(f"  Hard frames: {hard_indices}")
    print(f"  SD latents generated: {len(sd_latents)}")
    print(f"  Stats: {tgldp.get_stats()}")

    print("\nTest passed!")
