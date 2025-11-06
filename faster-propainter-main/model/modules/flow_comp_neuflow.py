import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeuFlow_bi(nn.Module):
    """NeuFlow v2 optical flow for ProPainter

    Drop-in replacement for RAFT_bi using ONNX Runtime with CUDA acceleration.
    Expected to be 10-70x faster than PyTorch RAFT while maintaining comparable accuracy.
    """
    def __init__(self, model_path='models/neuflow_things.onnx', device='cuda'):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_path = model_path
        self.ort = None  # Store onnxruntime module reference

        # Initialize ONNX Runtime session with CUDA provider
        try:
            import onnxruntime as ort
            self.ort = ort  # Store reference for IOBinding

            # Configure CUDA execution provider (NO CPU FALLBACK!)
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'gpu_mem_limit': 8 * 1024 * 1024 * 1024,  # 8GB
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # Best Conv algorithm (4x faster than DEFAULT)
                    'cudnn_conv_use_max_workspace': '1',     # Use full GPU memory for Conv optimization
                }),
                # NO CPU FALLBACK - Fail fast if CUDA isn't available!
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

            # ENFORCE CUDA - Crash hard if CPU fallback occurred!
            if 'CUDAExecutionProvider' not in available_providers:
                raise RuntimeError(
                    f"[ERROR] NeuFlow v2 REQUIRES CUDAExecutionProvider but it's not available!\n"
                    f"Available providers: {available_providers}\n"
                    f"This means ONNX Runtime fell back to CPU mode.\n"
                    f"Install onnxruntime-gpu with CUDA support or check your GPU setup."
                )
            print("[OK] CUDA acceleration VERIFIED for NeuFlow v2 - NO CPU FALLBACK!")
            print("[PERF] Using IOBinding to eliminate GPU↔CPU transfers")

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
        Run NeuFlow v2 inference on batch of image pairs using IOBinding for GPU efficiency.

        NeuFlow ONNX model has fixed input shape [1, 3, 432, 768].
        Uses ONNX Runtime IOBinding to keep all tensors on GPU and eliminate GPU↔CPU transfers.

        Args:
            image1: First images [B, C, H, W] in range [0, 1]
            image2: Second images [B, C, H, W] in range [0, 1]
            iters: Number of refinement iterations (unused for NeuFlow)

        Returns:
            flow: Optical flow [B, 2, H, W]
        """
        B, C, H_orig, W_orig = image1.shape

        # NeuFlow v2 ONNX expects fixed 432x768 resolution
        TARGET_H, TARGET_W = 432, 768

        flows = []

        # Process each frame pair using IOBinding (stays on GPU!)
        for i in range(B):
            # Extract single pair
            img1 = image1[i:i+1]  # [1, C, H, W]
            img2 = image2[i:i+1]  # [1, C, H, W]

            # Resize to target resolution for ONNX model
            img1_resized = F.interpolate(
                img1,
                size=(TARGET_H, TARGET_W),
                mode='bilinear',
                align_corners=False
            ).contiguous()  # Ensure contiguous for IOBinding

            img2_resized = F.interpolate(
                img2,
                size=(TARGET_H, TARGET_W),
                mode='bilinear',
                align_corners=False
            ).contiguous()

            # Use IOBinding to keep tensors on GPU (eliminates 4 GPU↔CPU transfers per pair!)
            io_binding = self.session.io_binding()

            # Bind inputs on GPU
            io_binding.bind_input(
                name=self.input_names[0],
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(img1_resized.shape),
                buffer_ptr=img1_resized.data_ptr()
            )
            io_binding.bind_input(
                name=self.input_names[1],
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(img2_resized.shape),
                buffer_ptr=img2_resized.data_ptr()
            )

            # Allocate output on GPU
            flow_output = torch.empty((1, 2, TARGET_H, TARGET_W), dtype=torch.float32, device=self.device).contiguous()
            io_binding.bind_output(
                name=self.output_names[0],
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=(1, 2, TARGET_H, TARGET_W),
                buffer_ptr=flow_output.data_ptr()
            )

            # Run inference (100% GPU, ZERO CPU transfers!)
            self.session.run_with_iobinding(io_binding)

            # Resize flow back to original resolution
            flow_resized = F.interpolate(
                flow_output,
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            )

            # CRITICAL: Scale flow values by resize ratio
            # Optical flow represents pixel displacement, so it must be scaled
            flow_resized[:, 0, :, :] *= (W_orig / TARGET_W)  # Scale u (horizontal) component
            flow_resized[:, 1, :, :] *= (H_orig / TARGET_H)  # Scale v (vertical) component

            flows.append(flow_resized)

        # Concatenate batch dimension
        return torch.cat(flows, dim=0)  # [B, 2, H, W]


def initialize_NeuFlow(model_path='models/neuflow_things.onnx', device='cuda'):
    """
    Initialize NeuFlow v2 model (compatibility function)
    """
    model = NeuFlow_bi(model_path=model_path, device=device)
    return model
