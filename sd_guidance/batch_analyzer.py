"""
Batch Difficulty Analyzer for TGLDP v2.0

Pre-scans ENTIRE video to identify the 15% hardest frames.
Instead of per-frame SD (slow), we batch-process strategically.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BatchDifficultyAnalyzer:
    """
    Pre-scans ENTIRE video to identify the hardest frames.
    Only these frames will get SD intervention - the rest use pure ProPainter.

    This is the KEY optimization: instead of running SD on every frame,
    we identify the 10-20% that actually NEED semantic hallucination.
    """

    def __init__(
        self,
        hard_frame_ratio: float = 0.15,
        min_hard_frames: int = 1,
        max_hard_frames: int = 50,
        device: str = "cuda",
    ):
        """
        Args:
            hard_frame_ratio: Fraction of frames to mark as "hard" (0.15 = 15%)
            min_hard_frames: Minimum number of hard frames to select
            max_hard_frames: Maximum number of hard frames (caps SD compute)
            device: Device for tensor operations
        """
        self.hard_frame_ratio = hard_frame_ratio
        self.min_hard_frames = min_hard_frames
        self.max_hard_frames = max_hard_frames
        self.device = device

    def analyze_video(
        self,
        frames: torch.Tensor,       # [T, C, H, W]
        masks: torch.Tensor,        # [T, 1, H, W]
        flow_maps: Optional[torch.Tensor] = None  # [T-1, 2, H, W]
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Analyze entire video and identify hard frames.

        Returns:
            - List of frame indices that need SD intervention
            - Difficulty scores for all frames [T]
        """
        logger.info("Analyzing video difficulty across all frames...")

        T = frames.shape[0]
        if frames.dim() == 4:
            _, C, H, W = frames.shape
        else:
            raise ValueError(f"Expected [T, C, H, W], got {frames.shape}")

        # Move to device if needed
        frames = frames.to(self.device)
        masks = masks.to(self.device)

        difficulty_scores = torch.zeros(T, device=self.device)

        for i in range(T):
            frame = frames[i]  # [C, H, W]
            mask = masks[i]    # [1, H, W]

            # Metric 1: Mask size (biggest factor - large masks need hallucination)
            mask_area = mask.sum() / (H * W)

            # Metric 2: Motion magnitude in masked region
            if i > 0:
                frame_diff = (frames[i] - frames[i-1]).abs().mean(dim=0)  # [H, W]
                motion_in_mask = (frame_diff * mask.squeeze()).sum() / (mask.sum() + 1e-6)
            else:
                motion_in_mask = torch.tensor(0.0, device=self.device)

            # Metric 3: Texture complexity (low texture = needs SD hallucination)
            gray_frame = frames[i].mean(dim=0)  # [H, W]

            # Sobel gradients for texture
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

            gray_4d = gray_frame.unsqueeze(0).unsqueeze(0)
            gx = torch.nn.functional.conv2d(gray_4d, sobel_x, padding=1).squeeze()
            gy = torch.nn.functional.conv2d(gray_4d, sobel_y, padding=1).squeeze()

            texture_mag = torch.sqrt(gx**2 + gy**2)
            # LOW texture in mask = HIGH difficulty (needs SD to hallucinate structure)
            texture_in_mask = (texture_mag * mask.squeeze()).sum() / (mask.sum() + 1e-6)
            texture_difficulty = 1.0 / (texture_in_mask + 0.1)  # Invert: low texture = high difficulty

            # Metric 4: Mask compactness (scattered masks are harder)
            # Use the ratio of mask area to bounding box area
            mask_2d = mask.squeeze()
            if mask_2d.sum() > 0:
                nonzero = torch.nonzero(mask_2d)
                if len(nonzero) > 0:
                    bbox_h = nonzero[:, 0].max() - nonzero[:, 0].min() + 1
                    bbox_w = nonzero[:, 1].max() - nonzero[:, 1].min() + 1
                    bbox_area = bbox_h * bbox_w
                    compactness = mask_2d.sum() / bbox_area
                    scatter_penalty = 1.0 - compactness  # Less compact = harder
                else:
                    scatter_penalty = torch.tensor(0.0, device=self.device)
            else:
                scatter_penalty = torch.tensor(0.0, device=self.device)

            # Combined difficulty score
            difficulty_scores[i] = (
                mask_area * 0.40 +              # 40% mask size
                motion_in_mask * 0.25 +         # 25% motion
                texture_difficulty * 0.20 +     # 20% texture (inverted)
                scatter_penalty * 0.15          # 15% scatter
            )

        # Normalize scores to [0, 1]
        if difficulty_scores.max() > difficulty_scores.min():
            difficulty_scores = (difficulty_scores - difficulty_scores.min()) / \
                               (difficulty_scores.max() - difficulty_scores.min())

        # Select top-K hardest frames
        k = int(T * self.hard_frame_ratio)
        k = max(self.min_hard_frames, min(k, self.max_hard_frames))

        # Get indices of hardest frames
        _, top_indices = torch.topk(difficulty_scores, k)
        hardest_indices = sorted(top_indices.cpu().tolist())

        logger.info(f"Identified {len(hardest_indices)} hard frames out of {T} total")
        logger.info(f"Hard frame indices: {hardest_indices}")
        logger.info(f"Difficulty range: {difficulty_scores.min():.4f} - {difficulty_scores.max():.4f}")

        return hardest_indices, difficulty_scores

    def get_keyframe_neighbors(
        self,
        hard_indices: List[int],
        total_frames: int,
        neighbor_radius: int = 2,
    ) -> List[int]:
        """
        Expand hard frames to include neighbors for smoother transitions.

        Args:
            hard_indices: List of hard frame indices
            total_frames: Total number of frames in video
            neighbor_radius: Number of frames to include on each side

        Returns:
            Expanded list of indices (sorted, unique)
        """
        expanded = set()
        for idx in hard_indices:
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                neighbor = idx + offset
                if 0 <= neighbor < total_frames:
                    expanded.add(neighbor)

        return sorted(list(expanded))


# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing BatchDifficultyAnalyzer...")

    # Create analyzer
    analyzer = BatchDifficultyAnalyzer(hard_frame_ratio=0.15)

    # Create dummy video data
    T, C, H, W = 100, 3, 480, 640
    frames = torch.rand(T, C, H, W, device="cuda")
    masks = torch.zeros(T, 1, H, W, device="cuda")

    # Create varying mask sizes (larger in middle of video)
    for i in range(T):
        center = T // 2
        dist = abs(i - center)
        size = max(50, 150 - dist * 2)
        y_start = H // 2 - size // 2
        x_start = W // 2 - size // 2
        masks[i, 0, y_start:y_start+size, x_start:x_start+size] = 1

    # Analyze
    hard_indices, scores = analyzer.analyze_video(frames, masks)

    print(f"\nResults:")
    print(f"  Total frames: {T}")
    print(f"  Hard frames: {len(hard_indices)}")
    print(f"  Hard indices: {hard_indices}")
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Test neighbor expansion
    expanded = analyzer.get_keyframe_neighbors(hard_indices, T, neighbor_radius=2)
    print(f"  Expanded indices: {len(expanded)} frames")

    print("\nTest passed!")
