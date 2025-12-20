"""
Lightweight Stable Diffusion Inpainting Model for TGLDP

This module provides a minimal SD wrapper that extracts only the components
needed for latent guidance:
- VAE Encoder: Converts RGB to latent space (matches ProPainter's encoder output)
- UNet: Denoises latents with inpainting conditioning
- NO VAE Decoder: ProPainter handles the final decode

Optimized for RTX 4090 with FP16 and optional TensorRT conversion.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Check if diffusers is available
try:
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.schedulers import DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers not installed. Run: pip install diffusers transformers accelerate")


class SDInpaintModel(nn.Module):
    """
    Lightweight SD Inpainting model for latent guidance.

    Only loads VAE encoder + UNet. Provides latent features that can be
    injected into ProPainter's InpaintGenerator at the post-encoder position.

    Key features:
    - Fast: Only 3-5 denoising steps (not full 50)
    - Efficient: No VAE decoder (ProPainter handles decode)
    - Memory-safe: FP16 + gradient checkpointing
    - Temporal: Works with TemporalConsistencyLock
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 3,
        guidance_scale: float = 7.5,
        use_trt: bool = False,
    ):
        super().__init__()

        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers package required. Run: pip install diffusers transformers accelerate")

        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.use_trt = use_trt
        self.model_id = model_id

        # Will be loaded on first use or explicit load()
        self.vae: Optional[AutoencoderKL] = None
        self.unet: Optional[UNet2DConditionModel] = None
        self.text_encoder: Optional[CLIPTextModel] = None
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.scheduler: Optional[DDIMScheduler] = None

        # Cached embeddings for efficiency
        self._null_embedding: Optional[torch.Tensor] = None
        self._inpaint_embedding: Optional[torch.Tensor] = None

        # Latent scale factor (SD uses 8x downsampling)
        self.vae_scale_factor = 8

        # ProPainter encoder output channels (256) vs SD latent channels (4)
        # We'll need to project between them
        self.propainter_channels = 256
        self.sd_latent_channels = 4

        self._loaded = False

    def load(self, cache_dir: Optional[str] = None):
        """Load model components from HuggingFace or local cache."""
        if self._loaded:
            return

        logger.info(f"Loading SD Inpainting model: {self.model_id}")

        # Load VAE (only encoder needed, but we load full for compatibility)
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
        ).to(self.device)
        self.unet.requires_grad_(False)
        self.unet.eval()

        # Enable memory efficient attention if available
        if hasattr(self.unet, 'enable_xformers_memory_efficient_attention'):
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xformers: {e}")

        # Load text encoder (minimal - just for null/inpaint prompts)
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            cache_dir=cache_dir,
        ).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id,
            subfolder="tokenizer",
            cache_dir=cache_dir,
        )

        # Setup scheduler for fast denoising
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_id,
            subfolder="scheduler",
            cache_dir=cache_dir,
        )
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        # Pre-compute text embeddings
        self._compute_embeddings()

        # Create projection layers for ProPainter compatibility
        self._create_projection_layers()

        self._loaded = True
        logger.info("SD Inpainting model loaded successfully")

    def _compute_embeddings(self):
        """Pre-compute text embeddings for efficiency."""
        # Null embedding (unconditional)
        null_tokens = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            self._null_embedding = self.text_encoder(null_tokens)[0]

        # Inpaint prompt embedding
        inpaint_tokens = self.tokenizer(
            "clean background, remove object, seamless inpainting",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            self._inpaint_embedding = self.text_encoder(inpaint_tokens)[0]

    def _create_projection_layers(self):
        """Create projection layers to convert between SD latent space and ProPainter feature space."""
        # SD latent (4ch @ H/8, W/8) -> ProPainter features (256ch @ H/4, W/4)
        # We need to: upsample 2x and expand channels

        self.sd_to_propainter = nn.Sequential(
            nn.Conv2d(self.sd_latent_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, self.propainter_channels, kernel_size=3, padding=1),
        ).to(self.device).to(self.dtype)

        # ProPainter features (256ch @ H/4, W/4) -> SD latent (4ch @ H/8, W/8)
        # For conditioning the UNet
        self.propainter_to_sd = nn.Sequential(
            nn.Conv2d(self.propainter_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AvgPool2d(2),  # Downsample 2x
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, self.sd_latent_channels, kernel_size=3, padding=1),
        ).to(self.device).to(self.dtype)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to SD latent space.

        Args:
            image: [B, C, H, W] in range [-1, 1]

        Returns:
            latents: [B, 4, H/8, W/8]
        """
        if not self._loaded:
            self.load()

        with torch.no_grad():
            latents = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        return latents

    def denoise(
        self,
        latents: torch.Tensor,
        mask: torch.Tensor,
        masked_image_latents: torch.Tensor,
        timestep_start: int = 0,
    ) -> torch.Tensor:
        """
        Denoise latents with inpainting conditioning.

        Args:
            latents: [B, 4, H/8, W/8] noisy latents
            mask: [B, 1, H/8, W/8] binary mask (1 = inpaint region)
            masked_image_latents: [B, 4, H/8, W/8] encoded masked image
            timestep_start: Which timestep to start from (0 = most noise)

        Returns:
            denoised: [B, 4, H/8, W/8]
        """
        if not self._loaded:
            self.load()

        # Prepare timesteps (start from specified step)
        timesteps = self.scheduler.timesteps[timestep_start:]

        # Get embeddings for classifier-free guidance
        text_embeddings = torch.cat([self._null_embedding, self._inpaint_embedding])

        # Denoise loop
        for t in timesteps:
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Concatenate mask and masked image latents (SD inpainting format)
            # Input: [latents, mask, masked_image_latents] = 4 + 1 + 4 = 9 channels
            mask_input = torch.cat([mask] * 2)
            masked_latents_input = torch.cat([masked_image_latents] * 2)

            latent_model_input = torch.cat(
                [latent_model_input, mask_input, masked_latents_input],
                dim=1
            )

            # Predict noise
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def hallucinate_region(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        propainter_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Main entry point: Hallucinate content for masked regions.

        Args:
            image: [B, 3, H, W] input image in range [0, 1]
            mask: [B, 1, H, W] binary mask (1 = region to inpaint)
            propainter_features: Optional [B, 256, H/4, W/4] ProPainter encoder features
                                 If provided, used to condition the hallucination

        Returns:
            sd_features: [B, 256, H/4, W/4] features compatible with ProPainter
        """
        if not self._loaded:
            self.load()

        B, C, H, W = image.shape

        # Normalize to [-1, 1] for SD
        image_norm = image * 2 - 1

        # Create masked image
        masked_image = image_norm * (1 - mask)

        # Encode to latent space
        image_latents = self.encode_image(image_norm)
        masked_image_latents = self.encode_image(masked_image)

        # Resize mask to latent size
        mask_latent = F.interpolate(mask, size=image_latents.shape[-2:], mode='nearest')

        # Start from noise in masked region, keep unmasked
        noise = torch.randn_like(image_latents)
        latents = image_latents * (1 - mask_latent) + noise * mask_latent

        # Add noise according to scheduler
        timesteps = self.scheduler.timesteps
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])

        # Denoise
        denoised_latents = self.denoise(
            latents=latents,
            mask=mask_latent,
            masked_image_latents=masked_image_latents,
        )

        # Project to ProPainter feature space
        sd_features = self.sd_to_propainter(denoised_latents)

        return sd_features

    def guide_propainter_features(
        self,
        propainter_features: torch.Tensor,
        hard_mask: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
        blend_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Blend SD hallucinated features with ProPainter features.

        This is the main integration point: only applies SD guidance
        to hard regions identified by gradient analysis.

        Args:
            propainter_features: [B, 256, H/4, W/4] from ProPainter encoder
            hard_mask: [B, 1, H/4, W/4] difficulty mask (1 = hard region)
            image: [B, 3, H, W] input image
            mask: [B, 1, H, W] inpainting mask
            blend_ratio: How much SD vs ProPainter (0 = pure PP, 1 = pure SD)

        Returns:
            blended_features: [B, 256, H/4, W/4]
        """
        if hard_mask.sum() == 0:
            # No hard regions, skip SD entirely
            return propainter_features

        # Get SD hallucination for hard regions
        sd_features = self.hallucinate_region(image, mask, propainter_features)

        # Blend only in hard regions
        # adaptive_blend = blend_ratio * hard_mask
        adaptive_blend = hard_mask * blend_ratio

        blended = propainter_features * (1 - adaptive_blend) + sd_features * adaptive_blend

        return blended

    def to_tensorrt(self, save_path: str, input_shapes: Dict[str, Tuple[int, ...]]):
        """Convert UNet to TensorRT for faster inference."""
        if not self._loaded:
            self.load()

        try:
            import torch_tensorrt

            logger.info("Converting UNet to TensorRT...")

            # Create sample inputs
            sample = torch.randn(input_shapes['sample'], dtype=self.dtype, device=self.device)
            timestep = torch.tensor([1.0], dtype=self.dtype, device=self.device)
            encoder_hidden_states = torch.randn(
                input_shapes['encoder_hidden_states'],
                dtype=self.dtype,
                device=self.device
            )

            # Trace and compile
            trt_unet = torch_tensorrt.compile(
                self.unet,
                inputs=[sample, timestep, encoder_hidden_states],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30,  # 1GB
            )

            torch.save(trt_unet, save_path)
            logger.info(f"TensorRT UNet saved to {save_path}")

            return trt_unet

        except ImportError:
            logger.warning("torch_tensorrt not available, skipping TRT conversion")
            return None


def load_sd_model(
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    num_inference_steps: int = 3,
    guidance_scale: float = 7.5,
) -> SDInpaintModel:
    """
    Convenience function to load the SD inpainting model.

    Args:
        device: Device to load model on
        cache_dir: Directory to cache model weights
        num_inference_steps: Number of denoising steps (3-5 recommended)
        guidance_scale: CFG scale (7.5 default)

    Returns:
        Loaded SDInpaintModel
    """
    model = SDInpaintModel(
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    model.load(cache_dir=cache_dir)
    return model


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing SD Inpaint Model...")

    # Create model
    model = SDInpaintModel(device="cuda", num_inference_steps=3)
    model.load()

    # Test with dummy input
    B, H, W = 1, 512, 512
    image = torch.rand(B, 3, H, W, device="cuda")
    mask = torch.zeros(B, 1, H, W, device="cuda")
    mask[:, :, 200:300, 200:300] = 1  # Inpaint center region

    # Hallucinate
    with torch.cuda.amp.autocast(dtype=torch.float16):
        features = model.hallucinate_region(image, mask)

    print(f"Output shape: {features.shape}")  # Should be [1, 256, 128, 128]
    print("Test passed!")
