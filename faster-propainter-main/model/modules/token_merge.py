"""
Token Merging for Sparse Temporal Transformer
Reduces spatial tokens (H*W) while preserving temporal structure (T)
Based on ToMe (Token Merging) by Meta, adapted for [B, T, H, W, C] format
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTokenMerger(nn.Module):
    """
    Merges similar spatial tokens within each frame
    Input: [B, T, H, W, C]
    Output: [B, T, H', W', C] where H'*W' < H*W
    """
    def __init__(self, merge_ratio=0.5):
        super().__init__()
        self.merge_ratio = merge_ratio  # Fraction of tokens to merge (0.5 = merge half)

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W, C] tensor
        Returns:
            merged_x: [B, T, H', W', C] tensor with reduced spatial dimensions
            unmerge_fn: Function to restore original shape (for skip connections)
        """
        B, T, H, W, C = x.shape
        N = H * W  # Total spatial tokens per frame

        # Reshape to [B*T, N, C] for processing each frame independently
        x_flat = x.view(B * T, N, C)

        # Compute number of tokens to merge
        num_to_merge = int(N * self.merge_ratio)
        num_keep = N - num_to_merge

        # Compute cosine similarity between all tokens
        x_norm = F.normalize(x_flat, p=2, dim=-1)  # [B*T, N, C]
        similarity = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B*T, N, N]

        # For each token, find its most similar token (excluding itself)
        similarity.fill_diagonal_(-float('inf'))  # Mask self-similarity
        most_similar = similarity.argmax(dim=-1)  # [B*T, N]

        # Select which tokens to merge based on similarity scores
        # Keep tokens with lowest average similarity (most unique)
        avg_similarity = similarity.mean(dim=-1)  # [B*T, N]
        _, keep_indices = torch.topk(avg_similarity, num_keep, largest=False, sorted=False)

        # Create merge indices: for each token, map it to either itself (if kept) or its merge target
        merge_indices = torch.arange(N, device=x.device).unsqueeze(0).expand(B * T, -1)

        # Tokens not in keep_indices get merged into their most similar token
        merge_mask = torch.ones(B * T, N, dtype=torch.bool, device=x.device)
        merge_mask.scatter_(1, keep_indices, False)

        merge_indices[merge_mask] = most_similar[merge_mask]

        # Merge tokens by averaging those assigned to the same index
        merged_x = torch.zeros(B * T, num_keep, C, device=x.device, dtype=x.dtype)
        merge_counts = torch.zeros(B * T, num_keep, 1, device=x.device, dtype=x.dtype)

        # Use scatter_add for efficient merging
        keep_indices_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, C)
        merged_x.scatter_add_(1, keep_indices_expanded, x_flat)

        keep_indices_counts = keep_indices.unsqueeze(-1)
        merge_counts.scatter_add_(1, keep_indices_counts, torch.ones(B * T, num_keep, 1, device=x.device))

        merged_x = merged_x / (merge_counts + 1e-6)  # Average

        # Reshape back to [B, T, H', W', C]
        # We reduce spatial dimensions approximately proportionally
        H_new = int(H * (num_keep / N) ** 0.5)
        W_new = num_keep // H_new

        # Pad if necessary
        if H_new * W_new < num_keep:
            W_new += 1
        if H_new * W_new > num_keep:
            # Pad merged_x
            pad_size = H_new * W_new - num_keep
            padding = torch.zeros(B * T, pad_size, C, device=x.device, dtype=x.dtype)
            merged_x = torch.cat([merged_x, padding], dim=1)

        merged_x = merged_x[:, :H_new * W_new].view(B, T, H_new, W_new, C)

        # Create unmerge function for restoring shape (if needed for skip connections)
        def unmerge_fn(merged_tokens):
            """Restore original spatial dimensions by nearest neighbor upsampling"""
            BM, TM, HM, WM, CM = merged_tokens.shape
            return F.interpolate(
                merged_tokens.permute(0, 1, 4, 2, 3).reshape(BM * TM, CM, HM, WM),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).reshape(BM, TM, CM, H, W).permute(0, 1, 3, 4, 2)

        return merged_x, unmerge_fn


class SimpleTokenMerger(nn.Module):
    """
    Simpler token merging using average pooling
    Much faster than similarity-based merging
    """
    def __init__(self, merge_ratio=0.5):
        super().__init__()
        self.merge_ratio = merge_ratio
        # Pool size to achieve target merge ratio (with proper rounding!)
        self.pool_size = int(1.0 / (1.0 - merge_ratio) ** 0.5 + 0.5)
        self._logged = False  # For debug logging

    def forward(self, x):
        """
        Args:
            x: [B, T, H, W, C]
        Returns:
            merged_x, unmerge_fn
        """
        B, T, H, W, C = x.shape

        # Use average pooling to merge spatial tokens
        # Reshape to [B*T, C, H, W] for pooling
        x_pooled = x.permute(0, 1, 4, 2, 3).reshape(B * T, C, H, W)

        # Apply adaptive average pooling
        H_new = H // self.pool_size
        W_new = W // self.pool_size

        merged = F.adaptive_avg_pool2d(x_pooled, (H_new, W_new))

        # Reshape back to [B, T, H', W', C]
        merged = merged.reshape(B, T, C, H_new, W_new).permute(0, 1, 3, 4, 2)

        # Debug logging (once per model instance)
        if not self._logged:
            reduction = (H * W) / (H_new * W_new)
            print(f"[TOKEN MERGE] {H}×{W} → {H_new}×{W_new} (pool_size={self.pool_size}, {reduction:.2f}x token reduction)")
            self._logged = True

        def unmerge_fn(merged_tokens):
            """Restore original size with bilinear upsampling"""
            BM, TM, HM, WM, CM = merged_tokens.shape
            upsampled = F.interpolate(
                merged_tokens.permute(0, 1, 4, 2, 3).reshape(BM * TM, CM, HM, WM),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            return upsampled.reshape(BM, TM, CM, H, W).permute(0, 1, 3, 4, 2)

        return merged, unmerge_fn
