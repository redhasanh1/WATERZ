"""
SD Guidance Module for TGLDP v2.0 (Temporal-Guided Latent Diffusion Propagation)

This module provides Stable Diffusion-based latent hallucination for
hard regions in video inpainting, with DEEP INJECTION into ProPainter's
encoder (not post-processing).

Architecture:
1. BatchDifficultyAnalyzer: Scan all frames, identify hardest 15%
2. SDPreprocessor: Pre-generate latents for ONLY hard frames
3. SDGuidedInpaintGenerator: Inject at encoder level

Usage:
    from sd_guidance import TGLDPv2Module

    tgldp = TGLDPv2Module(firstphoto_path="path/to/reference.jpg")
    hard_indices, sd_latents = tgldp.analyze_and_pregenerate(frames, masks)

    # Then use SDGuidedInpaintGenerator with sd_latents
"""

from .batch_analyzer import BatchDifficultyAnalyzer
from .preprocessor import SDPreprocessor
from .tgldp_integration import TGLDPv2Module, get_tgldp_v2, process_video_with_tgldp

# Optional imports (require diffusers)
try:
    from .sd_inpaint_model import SDInpaintModel, load_sd_model
    from .temporal_lock import TemporalConsistencyLock
except ImportError:
    SDInpaintModel = None
    load_sd_model = None
    TemporalConsistencyLock = None

__all__ = [
    'BatchDifficultyAnalyzer',
    'SDPreprocessor',
    'TGLDPv2Module',
    'get_tgldp_v2',
    'process_video_with_tgldp',
    'SDInpaintModel',
    'load_sd_model',
    'TemporalConsistencyLock',
]
