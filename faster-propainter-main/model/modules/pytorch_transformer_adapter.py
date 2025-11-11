"""
PyTorch Transformer Production Adapter

This adapter wraps the PyTorchTransformerWrapper to match the API expected by
the watermark removal pipeline, handling FP16 I/O and thread safety.

Usage in watermark.py:
    from model.modules.pytorch_transformer_adapter import PyTorchTransformerAdapter
    wrapper = PyTorchTransformerAdapter(checkpoint_path="weights/ProPainter.pth")
    model.transformers = wrapper  # Replace TRT adapter
"""
import torch
import torch.nn as nn
from .pytorch_transformer_wrapper import PyTorchTransformerWrapper


class PyTorchTransformerAdapter(nn.Module):
    """
    Production adapter for PyTorch transformer that matches _TransformerTRTAdapter API

    Key Features:
    - Handles FP16 input/output (converts to FP32 internally for accuracy)
    - Thread-safe via singleton pattern
    - Same API as TRT adapter: forward(x, fold_x_size, l_mask, t_dilation)
    - Dynamic shape support: [B, T, H, W, C] where T∈[2,16], H∈[6,15], W∈[9,15]
    """

    def __init__(self, checkpoint_path="weights/ProPainter.pth", device='cuda'):
        super().__init__()

        # Load the underlying PyTorch transformer wrapper
        self._transformer = PyTorchTransformerWrapper(checkpoint_path, device)
        self.device = device

        print(f"[PyTorch Adapter] Initialized transformer adapter")
        print(f"[PyTorch Adapter] Checkpoint: {checkpoint_path}")
        print(f"[PyTorch Adapter] Device: {device}")

    def forward(self, x, fold_x_size, l_mask=None, t_dilation=2):
        """
        Forward pass matching TRT adapter API

        Args:
            x: [B, T, H, W, C] feature tensor (typically FP16 in production)
            fold_x_size: (H_fold, W_fold) encoder feature map size
            l_mask: [B, T, H, W, 1] local mask (optional, defaults to all ones)
            t_dilation: temporal dilation factor (default: 2)

        Returns:
            output: [B, T, H, W, C] transformed features (same dtype as input)
        """
        B, T, H, W, C = x.shape
        input_dtype = x.dtype

        # Validate input shape (same constraints as TRT adapter)
        assert B == 1, f"Batch size must be 1, got {B}"
        assert C == 512, f"Channel count must be 512, got {C}"
        assert 2 <= T <= 16, f"Temporal frames must be in [2, 16], got {T}"
        assert 6 <= H <= 15, f"Height must be in [6, 15], got {H}"
        assert 9 <= W <= 15, f"Width must be in [9, 15], got {W}"

        # DEBUG: Log input/output ranges
        input_min = x.min().item()
        input_max = x.max().item()
        print(f"[ADAPTER] Input shape: {x.shape}, dtype: {input_dtype}")
        print(f"[ADAPTER] Input range: [{input_min:.3f}, {input_max:.3f}]")

        # Convert to FP32 for accuracy (transformer trained in FP32)
        if x.dtype == torch.float16:
            x = x.float()

        # Ensure contiguous memory layout
        x = x.contiguous()

        # Call underlying transformer
        with torch.no_grad():
            output = self._transformer(x, fold_x_size, l_mask, t_dilation)

        # DEBUG: Log output range before conversion
        output_min = output.min().item()
        output_max = output.max().item()
        print(f"[ADAPTER] Output (FP32) range: [{output_min:.3f}, {output_max:.3f}]")

        # Convert back to input dtype (typically FP16 in production)
        if input_dtype == torch.float16:
            output = output.half()
            print(f"[ADAPTER] Output (FP16) range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        return output

    def to(self, *args, **kwargs):
        """Override to() to handle device/dtype changes"""
        super().to(*args, **kwargs)
        self._transformer = self._transformer.to(*args, **kwargs)
        return self

    def half(self):
        """Override half() - we keep internal model as FP32, only I/O is FP16"""
        # Don't convert internal transformer to FP16 (it expects FP32)
        # Just mark that we accept FP16 input/output
        return self

    def eval(self):
        """Set to evaluation mode"""
        self._transformer.eval()
        return super().eval()


# Global singleton instance (thread-safe via GIL)
_global_adapter = None

def get_pytorch_adapter(checkpoint_path="weights/ProPainter.pth", device='cuda'):
    """
    Get or create global PyTorch transformer adapter (singleton pattern)

    This ensures we only load the model once across all Celery workers.
    """
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = PyTorchTransformerAdapter(checkpoint_path, device)
    return _global_adapter
