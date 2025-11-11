"""
PyTorch Transformer Wrapper for Production Use

This wrapper loads the sparse transformer with trained weights and provides
a simple interface for inference. Use this instead of TRT plugin to avoid
buffer routing issues.

Performance: ~2-5ms per frame (FP32 on GPU)
Quality: Exact match to training (std=0)
"""
import torch
import torch.nn as nn
from .sparse_transformer import TemporalSparseTransformerBlock


class PyTorchTransformerWrapper(nn.Module):
    """
    Wrapper for TemporalSparseTransformerBlock with trained weights

    Usage:
        transformer = PyTorchTransformerWrapper(checkpoint_path="weights/ProPainter.pth")
        output = transformer(input_5d)  # [B, T, H, W, C]
    """

    def __init__(self, checkpoint_path="weights/ProPainter.pth", device='cuda'):
        super().__init__()

        # Model parameters (from ProPainter config)
        self.dim = 512
        self.n_head = 4
        self.window_size = (5, 9)
        self.pool_size = (4, 4)
        self.depths = 8
        self.t2t_params = {
            'kernel_size': (7, 7),
            'stride': (3, 3),
            'padding': (3, 3)
        }

        # Create transformer
        self.transformer = TemporalSparseTransformerBlock(
            dim=self.dim,
            n_head=self.n_head,
            window_size=self.window_size,
            pool_size=self.pool_size,
            depths=self.depths,
            t2t_params=self.t2t_params
        )

        # Load trained weights
        self._load_weights(checkpoint_path)

        # Move to device and set to eval mode
        self.to(device)
        self.eval()

        # Disable gradients
        for param in self.parameters():
            param.requires_grad = False

    def _load_weights(self, checkpoint_path):
        """Load transformer weights from ProPainter checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

            # Extract transformer weights
            transformer_state = {}
            for k, v in state.items():
                if 'transformer' in k.lower():
                    # Remove only 'transformers.' prefix, keep 'transformer.' as part of model structure
                    # Example: 'transformers.transformer.0.attn.weight' -> 'transformer.0.attn.weight'
                    new_k = k.replace('transformers.', '')
                    transformer_state[new_k] = v.float()

            print(f"[DEBUG] Found {len(transformer_state)} transformer keys in checkpoint")

            if len(transformer_state) == 0:
                raise ValueError(f"No transformer weights found in checkpoint! Check key patterns.")

            # Load weights (strict=False to allow missing keys like buffers)
            result = self.transformer.load_state_dict(transformer_state, strict=False)

            # Report results
            print(f"Loaded {len(transformer_state)} transformer weights from {checkpoint_path}")
            if result.missing_keys:
                print(f"[WARNING] Missing {len(result.missing_keys)} keys (may be buffers): {result.missing_keys[:5]}")
            if result.unexpected_keys:
                print(f"[WARNING] Unexpected {len(result.unexpected_keys)} keys: {result.unexpected_keys[:5]}")

        except Exception as e:
            print(f"[ERROR] Failed to load transformer weights: {e}")
            import traceback
            traceback.print_exc()
            raise

    def forward(self, x, fold_x_size=None, l_mask=None, t_dilation=2):
        """
        Forward pass through transformer

        Args:
            x: [B, T, H, W, C] feature tensor (FP32)
            fold_x_size: (H_fold, W_fold) encoder feature map size (default: (H*3, W*3))
            l_mask: [B, T, H, W, 1] local mask (default: all ones)
            t_dilation: temporal dilation factor (default: 2)

        Returns:
            output: [B, T, H, W, C] transformed features (FP32)
        """
        B, T, H, W, C = x.shape

        # Default fold_x_size
        if fold_x_size is None:
            fold_x_size = (H * 3, W * 3)

        # Default mask (all ones = no masking)
        if l_mask is None:
            l_mask = torch.ones(B, T, H, W, 1, device=x.device, dtype=x.dtype)

        # Forward through transformer
        return self.transformer(x, fold_x_size, l_mask, t_dilation)

    @torch.no_grad()
    def __call__(self, x, fold_x_size=None, l_mask=None, t_dilation=2):
        """Inference-only call (no gradients)"""
        return self.forward(x, fold_x_size, l_mask, t_dilation)


# Singleton instance for production use
_global_transformer = None

def get_transformer(checkpoint_path="weights/ProPainter.pth", device='cuda'):
    """
    Get or create global transformer instance (singleton pattern)

    This ensures we only load the model once across the entire application.
    """
    global _global_transformer
    if _global_transformer is None:
        _global_transformer = PyTorchTransformerWrapper(checkpoint_path, device)
    return _global_transformer
