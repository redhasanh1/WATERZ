import os
import torch
import torch.nn as nn
import numpy as np


class NeuFlow_bi(nn.Module):
    """NeuFlow v2 optical flow for ProPainter

    Drop-in replacement for RAFT_bi using ONNX Runtime with CUDA acceleration.
    Expected to be 10-70x faster than PyTorch RAFT while maintaining comparable accuracy.
    """
    def __init__(self, model_path='models/neuflow_things.onnx', device='cuda'):
        super().__init__()
        self.device = device
        self.model_path = model_path

        # Initialize ONNX Runtime session with CUDA provider
        try:
            import onnxruntime as ort

            # Configure CUDA execution provider
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'DEFAULT',
                }),
                'CPUExecutionProvider'  # Fallback
            ]

            # Create session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output names
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]

            # Print provider info
            available_providers = self.session.get_providers()
            print(f"[OK] NeuFlow v2 loaded: {model_path}")
            print(f"[OK] ONNX Runtime providers: {available_providers}")

            # Check if CUDA is available
            if 'CUDAExecutionProvider' in available_providers:
                print("[OK] CUDA acceleration enabled for NeuFlow v2")
            else:
                print("[WARNING] CUDA not available, using CPU for NeuFlow v2")

        except ImportError:
            raise ImportError("onnxruntime-gpu is required. Install: pip install onnxruntime-gpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load NeuFlow v2 model: {e}")

        self.eval()  # Always in eval mode

    def forward(self, gt_local_frames, iters=8):
        """
        Compute bidirectional optical flow.

        Args:
            gt_local_frames: Input frames [B, T, C, H, W]
            iters: Number of refinement iterations (NeuFlow v2 uses 8)

        Returns:
            gt_flows_forward: Forward flows [B, T-1, 2, H, W]
            gt_flows_backward: Backward flows [B, T-1, 2, H, W]
        """
        b, l_t, c, h, w = gt_local_frames.size()

        with torch.no_grad():
            # Prepare frame pairs
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w).contiguous()
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w).contiguous()

            # Compute forward and backward flows
            gt_flows_forward = self._compute_flow(gtlf_1, gtlf_2, iters)
            gt_flows_backward = self._compute_flow(gtlf_2, gtlf_1, iters)

        # Reshape to [B, T-1, 2, H, W]
        gt_flows_forward = gt_flows_forward.view(b, l_t-1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t-1, 2, h, w)

        return gt_flows_forward, gt_flows_backward

    def _compute_flow(self, image1, image2, iters):
        """
        Run NeuFlow v2 inference on a pair of images.

        Args:
            image1: First image [B, C, H, W] in range [0, 1]
            image2: Second image [B, C, H, W] in range [0, 1]
            iters: Number of refinement iterations

        Returns:
            flow: Optical flow [B, 2, H, W]
        """
        # Convert to numpy and normalize if needed
        img1_np = image1.cpu().numpy().astype(np.float32)
        img2_np = image2.cpu().numpy().astype(np.float32)

        # Prepare inputs for ONNX model
        # Note: May need to adjust input format based on actual model requirements
        inputs = {
            self.input_names[0]: img1_np,
            self.input_names[1]: img2_np,
        }

        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        flow_np = outputs[0]  # Assuming first output is the flow

        # Convert back to PyTorch tensor
        flow_tensor = torch.from_numpy(flow_np).to(self.device)

        return flow_tensor


def initialize_NeuFlow(model_path='models/neuflow_things.onnx', device='cuda'):
    """
    Initialize NeuFlow v2 model (compatibility function)
    """
    model = NeuFlow_bi(model_path=model_path, device=device)
    return model
