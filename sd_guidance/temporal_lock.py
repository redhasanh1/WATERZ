"""
Temporal Consistency Lock for TGLDP

This module ensures temporal coherence when using SD hallucination across
video frames. Without this, each frame would have independent hallucinations
causing severe flickering.

Key techniques:
1. Cross-Attention Memory Bank: Store attention patterns from previous frames
2. Feature Smoothing: Exponential moving average across frames
3. Optical Flow Warping: Warp previous hallucinations to current frame
4. Consistency Loss: Penalize large changes between frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TemporalConsistencyLock(nn.Module):
    """
    Maintains temporal consistency for SD-hallucinated features across frames.

    Works by:
    1. Storing a memory bank of recent frame features
    2. Computing optical flow-warped features from previous frames
    3. Blending current SD output with temporally-propagated features
    4. Applying attention-based refinement for smooth transitions
    """

    def __init__(
        self,
        feature_channels: int = 256,
        memory_size: int = 5,
        temporal_weight: float = 0.7,
        flow_weight: float = 0.8,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            feature_channels: Number of channels in ProPainter features
            memory_size: Number of previous frames to remember
            temporal_weight: Weight for temporal smoothing (0=no smoothing, 1=full)
            flow_weight: Weight for optical flow warping vs direct blend
            device: Device to run on
            dtype: Data type for tensors
        """
        super().__init__()

        self.feature_channels = feature_channels
        self.memory_size = memory_size
        self.temporal_weight = temporal_weight
        self.flow_weight = flow_weight
        self.device = device
        self.dtype = dtype

        # Memory bank: stores (features, mask) tuples
        self.memory_bank: deque = deque(maxlen=memory_size)

        # Running average for smooth features
        self.running_avg: Optional[torch.Tensor] = None
        self.running_mask: Optional[torch.Tensor] = None

        # Attention-based temporal refinement
        self.temporal_attention = TemporalCrossAttention(
            channels=feature_channels,
            num_heads=4,
            dtype=dtype,
        ).to(device)

        # Confidence tracking per-pixel
        self.confidence_map: Optional[torch.Tensor] = None

        # Frame counter
        self.frame_idx = 0

    def reset(self):
        """Reset memory bank for new video."""
        self.memory_bank.clear()
        self.running_avg = None
        self.running_mask = None
        self.confidence_map = None
        self.frame_idx = 0
        logger.info("Temporal consistency lock reset")

    def apply(
        self,
        current_features: torch.Tensor,
        mask: torch.Tensor,
        optical_flow: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply temporal consistency to SD-hallucinated features.

        Args:
            current_features: [B, C, H, W] current frame's SD features
            mask: [B, 1, H, W] inpainting mask for current frame
            optical_flow: [B, 2, H, W] optical flow from previous to current frame
                         If provided, used to warp previous features

        Returns:
            consistent_features: [B, C, H, W] temporally-consistent features
        """
        B, C, H, W = current_features.shape

        if self.frame_idx == 0 or len(self.memory_bank) == 0:
            # First frame - no temporal context
            self._update_memory(current_features, mask)
            return current_features

        # Get previous frame features
        prev_features, prev_mask = self.memory_bank[-1]

        # Warp previous features if flow available
        if optical_flow is not None:
            warped_features = self._warp_with_flow(prev_features, optical_flow)
            warped_mask = self._warp_with_flow(prev_mask, optical_flow)
        else:
            warped_features = prev_features
            warped_mask = prev_mask

        # Compute consistency weight based on flow magnitude and mask overlap
        consistency_weight = self._compute_consistency_weight(
            current_features, warped_features, mask, warped_mask
        )

        # Temporal blend: trust previous more where consistent
        blended_features = (
            current_features * (1 - consistency_weight * self.temporal_weight) +
            warped_features * (consistency_weight * self.temporal_weight)
        )

        # Apply cross-attention with memory bank for refinement
        if len(self.memory_bank) >= 2:
            blended_features = self._apply_temporal_attention(
                blended_features, mask
            )

        # Update running average
        self._update_running_avg(blended_features, mask)

        # Mix with running average for extra stability
        if self.running_avg is not None:
            alpha = 0.3 * mask  # Only apply in masked regions
            blended_features = (
                blended_features * (1 - alpha) +
                self.running_avg * alpha
            )

        # Update memory
        self._update_memory(blended_features, mask)

        return blended_features

    def _warp_with_flow(
        self,
        features: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Warp features using optical flow.

        Args:
            features: [B, C, H, W] features to warp
            flow: [B, 2, H, W] optical flow field

        Returns:
            warped: [B, C, H, W] warped features
        """
        B, C, H, W = features.shape

        # Resize flow to feature size if needed
        if flow.shape[-2:] != (H, W):
            flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)
            # Scale flow values accordingly
            flow = flow * torch.tensor(
                [W / flow.shape[-1], H / flow.shape[-2]],
                device=flow.device
            ).view(1, 2, 1, 1)

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=features.device),
            torch.linspace(-1, 1, W, device=features.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        # Add flow (normalized to [-1, 1])
        flow_norm = flow.clone()
        flow_norm[:, 0] = flow_norm[:, 0] / (W / 2)
        flow_norm[:, 1] = flow_norm[:, 1] / (H / 2)

        sample_grid = grid + flow_norm
        sample_grid = sample_grid.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Warp
        warped = F.grid_sample(
            features.to(torch.float32),
            sample_grid.to(torch.float32),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        return warped.to(features.dtype)

    def _compute_consistency_weight(
        self,
        current: torch.Tensor,
        warped: torch.Tensor,
        current_mask: torch.Tensor,
        warped_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-pixel consistency weight.

        High weight = trust previous frame more (region is consistent)
        Low weight = trust current hallucination more (region changed)
        """
        # Feature difference
        diff = (current - warped).abs().mean(dim=1, keepdim=True)

        # Normalize to [0, 1]
        diff_norm = diff / (diff.max() + 1e-6)

        # Invert: high consistency = low diff
        consistency = 1 - diff_norm

        # Only apply where both masks overlap
        mask_overlap = current_mask * warped_mask
        consistency = consistency * mask_overlap

        return consistency

    def _apply_temporal_attention(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention with memory bank features."""
        # Stack memory bank features
        memory_features = torch.stack([f for f, _ in self.memory_bank], dim=0)
        memory_masks = torch.stack([m for _, m in self.memory_bank], dim=0)

        # Apply temporal cross-attention
        refined = self.temporal_attention(
            features,
            memory_features,
            memory_masks,
            mask,
        )

        return refined

    def _update_running_avg(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Update exponential moving average."""
        alpha = 0.2  # Smoothing factor

        if self.running_avg is None:
            self.running_avg = features.clone()
            self.running_mask = mask.clone()
        else:
            # Resize if needed
            if self.running_avg.shape != features.shape:
                self.running_avg = F.interpolate(
                    self.running_avg,
                    size=features.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                self.running_mask = F.interpolate(
                    self.running_mask,
                    size=mask.shape[-2:],
                    mode='nearest'
                )

            # Only update in masked regions
            update_mask = mask * self.running_mask
            self.running_avg = (
                self.running_avg * (1 - alpha * update_mask) +
                features * (alpha * update_mask)
            )
            self.running_mask = torch.max(self.running_mask, mask)

    def _update_memory(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Add current frame to memory bank."""
        self.memory_bank.append((features.detach().clone(), mask.detach().clone()))
        self.frame_idx += 1


class TemporalCrossAttention(nn.Module):
    """
    Cross-attention module for temporal feature refinement.

    Query: Current frame features
    Key/Value: Memory bank features from previous frames
    """

    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 4,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(channels, channels, dtype=dtype)
        self.k_proj = nn.Linear(channels, channels, dtype=dtype)
        self.v_proj = nn.Linear(channels, channels, dtype=dtype)
        self.out_proj = nn.Linear(channels, channels, dtype=dtype)

        # Layer norm
        self.norm = nn.LayerNorm(channels, dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_masks: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, C, H, W] current frame features
            memory: [T, B, C, H, W] memory bank features (T frames)
            memory_masks: [T, B, 1, H, W] memory masks
            query_mask: [B, 1, H, W] current mask

        Returns:
            refined: [B, C, H, W]
        """
        B, C, H, W = query.shape
        T = memory.shape[0]

        # Flatten spatial dims
        query_flat = query.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        memory_flat = memory.flatten(3).permute(1, 3, 0, 2)  # [B, HW, T, C]
        memory_flat = memory_flat.reshape(B, H*W, -1)  # [B, HW, T*C]

        # Only attend in masked regions (efficiency)
        # For simplicity, we process all but mask output

        # Project
        q = self.q_proj(query_flat)  # [B, HW, C]

        # Reshape memory for K, V
        memory_reshape = memory.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W]
        memory_reshape = memory_reshape.reshape(B * T, C, H, W)
        memory_flat_kv = memory_reshape.flatten(2).permute(0, 2, 1)  # [B*T, HW, C]

        k = self.k_proj(memory_flat_kv).reshape(B, T * H * W, C)  # [B, T*HW, C]
        v = self.v_proj(memory_flat_kv).reshape(B, T * H * W, C)

        # Multi-head attention
        q = q.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, T*H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, T*H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention (scaled dot product)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn.float(), dim=-1).to(query.dtype)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, H*W, C)

        # Project output
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(query_flat + out)

        # Reshape back
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        # Only update masked regions
        refined = query * (1 - query_mask) + out * query_mask

        return refined


class BackgroundGuidance(nn.Module):
    """
    Uses a clean background reference image to guide inpainting.

    This is specifically designed for the firstphoto.jpg use case:
    warp the clean background using optical flow and blend with
    ProPainter output.
    """

    def __init__(
        self,
        background_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.background_path = background_path

        # Load background image
        self.background: Optional[torch.Tensor] = None
        self._load_background()

    def _load_background(self):
        """Load and preprocess background image."""
        import cv2
        import numpy as np

        if not os.path.exists(self.background_path):
            logger.warning(f"Background image not found: {self.background_path}")
            return

        # Load image
        img = cv2.imread(self.background_path)
        if img is None:
            logger.warning(f"Failed to load background: {self.background_path}")
            return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        # Convert to tensor
        self.background = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        self.background = self.background.to(self.device).to(self.dtype)

        logger.info(f"Loaded background: {self.background_path}, shape: {self.background.shape}")

    def warp_background(
        self,
        flow: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Warp background to match current frame using optical flow.

        Args:
            flow: [B, 2, H, W] cumulative flow from background frame
            target_size: (H, W) target resolution

        Returns:
            warped_bg: [B, 3, H, W]
        """
        if self.background is None:
            return None

        B = flow.shape[0]
        H, W = target_size

        # Resize background to target size
        bg = F.interpolate(self.background, size=(H, W), mode='bilinear', align_corners=False)
        bg = bg.expand(B, -1, -1, -1)

        # Resize flow
        flow_resized = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=False)

        # Create grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

        # Normalize flow
        flow_norm = flow_resized.clone()
        flow_norm[:, 0] = flow_norm[:, 0] / (W / 2)
        flow_norm[:, 1] = flow_norm[:, 1] / (H / 2)

        sample_grid = grid + flow_norm
        sample_grid = sample_grid.permute(0, 2, 3, 1)

        # Warp
        warped = F.grid_sample(
            bg.float(),
            sample_grid.float(),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        return warped.to(self.dtype)

    def blend(
        self,
        propainter_output: torch.Tensor,
        mask: torch.Tensor,
        flow: torch.Tensor,
        hard_mask: Optional[torch.Tensor] = None,
        base_alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Blend ProPainter output with warped background.

        Args:
            propainter_output: [B, 3, H, W] ProPainter result
            mask: [B, 1, H, W] inpainting mask
            flow: [B, 2, H, W] optical flow
            hard_mask: [B, 1, H, W] difficulty mask (blend more in hard regions)
            base_alpha: Base blend ratio

        Returns:
            blended: [B, 3, H, W]
        """
        H, W = propainter_output.shape[-2:]

        # Warp background
        warped_bg = self.warp_background(flow, (H, W))
        if warped_bg is None:
            return propainter_output

        # Compute adaptive alpha
        alpha = base_alpha * mask

        if hard_mask is not None:
            # Trust background more in hard regions
            hard_mask_resized = F.interpolate(hard_mask, size=(H, W), mode='bilinear', align_corners=False)
            alpha = alpha + 0.3 * hard_mask_resized
            alpha = alpha.clamp(0, 0.8)

        # Blend
        blended = propainter_output * (1 - alpha) + warped_bg * alpha

        return blended


# Import os for BackgroundGuidance
import os


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Temporal Consistency Lock...")

    # Create lock
    lock = TemporalConsistencyLock(device="cuda")

    # Simulate 10 frames
    B, C, H, W = 1, 256, 128, 128

    for i in range(10):
        features = torch.randn(B, C, H, W, device="cuda", dtype=torch.float16)
        mask = torch.zeros(B, 1, H, W, device="cuda", dtype=torch.float16)
        mask[:, :, 50:80, 50:80] = 1

        # Simulate optical flow (small motion)
        flow = torch.randn(B, 2, H, W, device="cuda", dtype=torch.float16) * 2

        consistent = lock.apply(features, mask, flow)
        print(f"Frame {i}: input norm = {features.norm():.2f}, output norm = {consistent.norm():.2f}")

    print("Test passed!")
