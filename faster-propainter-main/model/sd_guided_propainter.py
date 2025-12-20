"""
SD-Guided ProPainter for TGLDP v2.0

This module provides a modified InpaintGenerator that accepts SD latent injection
at the encoder level. Instead of post-processing (which doesn't work), we inject
SD guidance directly into ProPainter's feature pipeline.

Key insight: ProPainter's encoder outputs 128-channel features at H/4, W/4.
SD latents are 4-channel at H/8, W/8. We need to:
1. Upsample SD latents to H/4, W/4
2. Project from 4 channels to 128 channels
3. Blend with encoder features based on difficulty mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

# Import base ProPainter
from .propainter import InpaintGenerator as BaseInpaintGenerator, Encoder

logger = logging.getLogger(__name__)


class SDLatentProjector(nn.Module):
    """
    Projects SD latents (4ch @ H/8) to ProPainter feature space (128ch @ H/4).

    This is the bridge between Stable Diffusion's latent space and
    ProPainter's encoder output space.
    """

    def __init__(self, sd_channels: int = 4, pp_channels: int = 128):
        super().__init__()

        self.projection = nn.Sequential(
            # Upsample 2x (H/8 -> H/4)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # Expand channels: 4 -> 32
            nn.Conv2d(sd_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 128 (match ProPainter)
            nn.Conv2d(64, pp_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Initialize with small weights for stable training
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sd_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sd_latent: [B, 4, H/8, W/8] SD latent

        Returns:
            projected: [B, 128, H/4, W/4] ProPainter-compatible features
        """
        return self.projection(sd_latent)


class SDGuidedEncoder(nn.Module):
    """
    Wraps ProPainter's encoder with SD latent injection capability.

    Injection strategy:
    1. Run normal encoder forward
    2. For frames with SD latents, project SD latent to feature space
    3. Blend based on mask: enc_feat = (1-α)*enc_feat + α*sd_feat
    """

    def __init__(self, base_encoder: Encoder):
        super().__init__()
        self.base_encoder = base_encoder
        self.sd_projector = SDLatentProjector(sd_channels=4, pp_channels=128)

        # Learnable blend weight (can be made adaptive)
        self.blend_weight = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        x: torch.Tensor,                    # [B*T, 5, H, W]
        sd_latents: Optional[Dict[int, torch.Tensor]] = None,
        frame_indices: Optional[List[int]] = None,
        masks: Optional[torch.Tensor] = None,  # [B*T, 1, H/4, W/4]
    ) -> torch.Tensor:
        """
        Forward with optional SD injection.

        Args:
            x: Encoder input (masked frames + masks)
            sd_latents: {frame_idx: latent_tensor} for hard frames
            frame_indices: Which frames in the batch correspond to which indices
            masks: Downsampled masks for weighted blending

        Returns:
            enc_feat: [B*T, 128, H/4, W/4]
        """
        # Base encoder forward
        enc_feat = self.base_encoder(x)

        # If no SD latents, return as-is
        if sd_latents is None or frame_indices is None:
            return enc_feat

        BT, C, H, W = enc_feat.shape

        # Inject SD latents for hard frames
        for batch_idx, frame_idx in enumerate(frame_indices):
            if frame_idx not in sd_latents:
                continue

            sd_latent = sd_latents[frame_idx]

            # Project SD latent to feature space
            sd_feat = self.sd_projector(sd_latent)

            # Resize if needed
            if sd_feat.shape[-2:] != (H, W):
                sd_feat = F.interpolate(sd_feat, size=(H, W), mode='bilinear', align_corners=False)

            # Get mask for this frame (if available)
            if masks is not None and batch_idx < masks.shape[0]:
                mask = masks[batch_idx:batch_idx+1]
                if mask.shape[-2:] != (H, W):
                    mask = F.interpolate(mask, size=(H, W), mode='nearest')
                # Blend only in masked region
                alpha = self.blend_weight.clamp(0.1, 0.9) * mask
            else:
                # Uniform blend
                alpha = self.blend_weight.clamp(0.1, 0.9)

            # Blend: SD features in masked region
            enc_feat[batch_idx:batch_idx+1] = (
                enc_feat[batch_idx:batch_idx+1] * (1 - alpha) +
                sd_feat * alpha
            )

            logger.debug(f"Injected SD latent at frame {frame_idx}")

        return enc_feat


class SDGuidedInpaintGenerator(BaseInpaintGenerator):
    """
    ProPainter's InpaintGenerator with deep SD latent injection.

    This replaces the standard encoder with SDGuidedEncoder, allowing
    SD semantic features to be injected at the encoder output level.

    Usage:
        model = SDGuidedInpaintGenerator(model_path="weights/propainter.pth")

        # During inference:
        output = model(
            masked_frames,
            flows,
            masks_in,
            masks_updated,
            num_local_frames,
            sd_latents={5: latent1, 10: latent2},  # Only hard frames
            hard_frame_indices=[5, 10],
        )
    """

    def __init__(self, init_weights=True, model_path=None):
        # Initialize base class (loads pretrained weights)
        super().__init__(init_weights=init_weights, model_path=model_path)

        # Wrap encoder with SD guidance capability
        self.sd_guided_encoder = SDGuidedEncoder(self.encoder)

        # Track injection stats
        self._injection_count = 0

        logger.info("SDGuidedInpaintGenerator initialized with latent injection")

    def forward(
        self,
        masked_frames: torch.Tensor,
        completed_flows: Tuple[torch.Tensor, torch.Tensor],
        masks_in: torch.Tensor,
        masks_updated: torch.Tensor,
        num_local_frames: int,
        interpolation: str = 'bilinear',
        t_dilation: int = 2,
        # NEW: SD guidance parameters
        sd_latents: Optional[Dict[int, torch.Tensor]] = None,
        hard_frame_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional SD latent injection.

        Args:
            masked_frames: [B, T, 3, H, W] input frames
            completed_flows: (forward, backward) optical flow
            masks_in: [B, T, 1, H, W] original masks
            masks_updated: [B, T, 1, H, W] updated masks
            num_local_frames: Number of local frames for processing
            interpolation: Interpolation mode
            t_dilation: Temporal dilation for transformer
            sd_latents: {frame_idx: latent} for hard frames (NEW)
            hard_frame_indices: List of frame indices with SD latents (NEW)

        Returns:
            output: [B, T, 3, H, W] inpainted frames
        """
        l_t = num_local_frames
        b, t, _, ori_h, ori_w = masked_frames.size()

        # Prepare encoder input
        encoder_input = torch.cat([
            masked_frames.view(b * t, 3, ori_h, ori_w),
            masks_in.view(b * t, 1, ori_h, ori_w),
            masks_updated.view(b * t, 1, ori_h, ori_w)
        ], dim=1)

        # Build frame index mapping for batch
        if hard_frame_indices is not None:
            # Map batch position to frame index
            frame_indices = list(range(t))  # Assuming sequential frames
        else:
            frame_indices = None

        # Downsample masks for blending
        masks_ds = F.interpolate(
            masks_in.view(b * t, 1, ori_h, ori_w),
            scale_factor=0.25,
            mode='nearest'
        )

        # === INJECTION POINT: SD-guided encoding ===
        enc_feat = self.sd_guided_encoder(
            encoder_input,
            sd_latents=sd_latents,
            frame_indices=frame_indices,
            masks=masks_ds,
        )

        # Track injections
        if sd_latents is not None:
            self._injection_count += len(sd_latents)

        # Rest of forward pass (unchanged from base class)
        _, c, h, w = enc_feat.size()
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]
        fold_feat_size = (h, w)

        ds_flows_f = F.interpolate(
            completed_flows[0].view(-1, 2, ori_h, ori_w),
            scale_factor=1/4,
            mode='bilinear',
            align_corners=False
        ).view(b, l_t-1, 2, h, w) / 4.0

        ds_flows_b = F.interpolate(
            completed_flows[1].view(-1, 2, ori_h, ori_w),
            scale_factor=1/4,
            mode='bilinear',
            align_corners=False
        ).view(b, l_t-1, 2, h, w) / 4.0

        ds_mask_in = F.interpolate(
            masks_in.reshape(-1, 1, ori_h, ori_w),
            scale_factor=1/4,
            mode='nearest'
        ).view(b, t, 1, h, w)

        ds_mask_in_local = ds_mask_in[:, :l_t]
        ds_mask_updated_local = F.interpolate(
            masks_updated[:, :l_t].reshape(-1, 1, ori_h, ori_w),
            scale_factor=1/4,
            mode='nearest'
        ).view(b, l_t, 1, h, w)

        if self.training:
            mask_pool_l = self.max_pool(ds_mask_in.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))
        else:
            mask_pool_l = self.max_pool(ds_mask_in_local.view(-1, 1, h, w))
            mask_pool_l = mask_pool_l.view(b, l_t, 1, mask_pool_l.size(-2), mask_pool_l.size(-1))

        prop_mask_in = torch.cat([ds_mask_in_local, ds_mask_updated_local], dim=2)
        _, _, local_feat, _ = self.feat_prop_module(local_feat, ds_flows_f, ds_flows_b, prop_mask_in, interpolation)
        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        from einops import rearrange
        trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_feat_size)
        mask_pool_l = rearrange(mask_pool_l, 'b t c h w -> b t h w c').contiguous()
        trans_feat = self.transformers(trans_feat, fold_feat_size, mask_pool_l, t_dilation=t_dilation)
        trans_feat = self.sc(trans_feat, t, fold_feat_size)
        trans_feat = trans_feat.view(b, t, -1, h, w)

        enc_feat = enc_feat + trans_feat

        if self.training:
            output = self.decoder(enc_feat.view(-1, c, h, w))
            output = torch.tanh(output).view(b, t, 3, ori_h, ori_w)
        else:
            output = self.decoder(enc_feat[:, :l_t].view(-1, c, h, w))
            output = torch.tanh(output).view(b, l_t, 3, ori_h, ori_w)

        return output

    def get_injection_stats(self) -> Dict[str, int]:
        """Get statistics about SD injections."""
        return {
            'total_injections': self._injection_count,
        }

    def reset_stats(self):
        """Reset injection statistics."""
        self._injection_count = 0


def load_sd_guided_propainter(
    model_path: str = None,
    device: str = "cuda",
    use_half: bool = True,
) -> SDGuidedInpaintGenerator:
    """
    Load SD-guided ProPainter model.

    Args:
        model_path: Path to ProPainter weights
        device: Device to load model on
        use_half: Whether to use FP16

    Returns:
        Loaded model
    """
    model = SDGuidedInpaintGenerator(model_path=model_path)
    model = model.to(device)

    if use_half:
        model = model.half()

    model.eval()

    logger.info(f"Loaded SDGuidedInpaintGenerator on {device}")
    return model


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing SDGuidedInpaintGenerator...")

    # Test SD projector
    projector = SDLatentProjector().cuda().half()
    sd_latent = torch.randn(1, 4, 60, 80, device="cuda", dtype=torch.float16)
    projected = projector(sd_latent)
    print(f"SD Projector: {sd_latent.shape} -> {projected.shape}")

    print("\nTest passed!")
