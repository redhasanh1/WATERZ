"""
NVDEC GPU-Only Video Loader
Keeps frames on GPU throughout entire pipeline - NO CPU transfers!

Eliminates the bottleneck from FINAL_NVDEC_TEST.py line 64:
    OLD: frame_np = torch_tensor.cpu().numpy()  # GPU→CPU transfer!
    NEW: Keep torch_tensor on GPU → direct ProPainter input
"""
import os
import torch
import PyNvVideoCodec as nvc


class NVDECVideoLoader:
    """GPU-accelerated video decoder using NVIDIA NVDEC.

    Outputs CUDA tensors directly ready for ProPainter - NO CPU transfers!
    """

    def __init__(self, video_path, device_id=0):
        """Initialize NVDEC decoder.

        Args:
            video_path: Path to video file
            device_id: GPU device ID (default: 0)
        """
        self.video_path = video_path
        self.device_id = device_id

        # Create demuxer
        self.demuxer = nvc.CreateDemuxer(filename=video_path)

        # Get codec and create decoder with RGB output
        codec = self.demuxer.GetNvCodecId()
        self.decoder = nvc.CreateDecoder(
            gpuid=device_id,
            codec=codec,
            outputColorType=nvc.OutputColorType.RGB  # Decode directly to RGB
        )

        # Get video properties (PyNvVideoCodec API: FrameRate(), Width(), Height())
        self.fps = self.demuxer.FrameRate()
        self.width = self.demuxer.Width()
        self.height = self.demuxer.Height()

        # Total frames not available upfront - count during decode
        self.total_frames = None

    def load_all_frames_gpu(self, normalize=True, target_device='cuda:0'):
        """Load all video frames directly to GPU tensors (ZERO CPU TRANSFERS!).

        Args:
            normalize: If True, normalize to [-1, 1] range (ProPainter format)
            target_device: Target CUDA device (default: 'cuda:0')

        Returns:
            torch.Tensor: Shape [1, T, C, H, W], dtype=float32, range=[-1,1]
                         Ready for direct input to ProPainter model!
        """
        frames_gpu = []
        packet_count = 0

        while packet_count < 10000:  # Safety limit
            try:
                # Demux packet
                packet = self.demuxer.Demux()
                if packet is None:
                    break

                # Decode packet (outputs to GPU memory)
                decoded = self.decoder.Decode(packet)

                if decoded and len(decoded) > 0:
                    for frame in decoded:
                        # DLPack → PyTorch tensor (ZERO-COPY, stays on GPU!)
                        torch_tensor = torch.from_dlpack(frame)

                        # CRITICAL: Clone tensor to avoid buffer reuse!
                        # NVDEC reuses same GPU buffer - must copy data
                        frames_gpu.append(torch_tensor.clone())

                packet_count += 1

            except Exception as e:
                print(f"[NVDEC] Decode error at packet {packet_count}: {e}")
                break

        if len(frames_gpu) == 0:
            raise RuntimeError(f"[NVDEC] No frames decoded from {self.video_path}")

        # Update total_frames after counting during decode
        self.total_frames = len(frames_gpu)

        # Stack frames into single tensor [T, H, W, C]
        frames_batch = torch.stack(frames_gpu, dim=0)

        # FIX 4: Sync after stack
        torch.cuda.synchronize()

        print(f"[NVDEC] Decoded {len(frames_gpu)} frames to GPU (shape: {frames_batch.shape})")

        # Reshape from [T, H, W, C] to [1, T, C, H, W] first
        # T=temporal, C=channels (RGB), H=height, W=width
        # FIX 3: Make contiguous after permute
        frames_reshaped = frames_batch.permute(0, 3, 1, 2).contiguous().unsqueeze(0)

        # Convert to float32 for processing
        frames_float = frames_reshaped.float()

        if normalize:
            # Normalize from [0, 255] to [-1, 1] (ProPainter format)
            frames_float = (frames_float / 255.0) * 2.0 - 1.0

        # FIX 5: Remove non_blocking for safe transfer
        frames_final = frames_float.to(target_device)

        print(f"[NVDEC] Final tensor shape: {frames_final.shape}, "
              f"range: [{frames_final.min():.2f}, {frames_final.max():.2f}], "
              f"device: {frames_final.device}, "
              f"normalized: {normalize}")

        return frames_final

    def get_video_info(self):
        """Get video metadata.

        Returns:
            dict: Video properties (fps, total_frames, width, height)
        """
        return {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'width': self.width,
            'height': self.height
        }

    def get_properties(self):
        """Get video properties (legacy compatibility).

        Returns:
            dict: Video properties for server_production.py compatibility
        """
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames
        }

    def load_all_frames(self, to_numpy=False, color_format='RGB'):
        """Load frames (legacy compatibility method).

        For backwards compatibility with existing code. If to_numpy=True,
        converts GPU tensors to numpy arrays (loses GPU acceleration benefit).

        Args:
            to_numpy: If True, convert to numpy arrays (CPU transfer!)
            color_format: 'RGB' or 'BGR'

        Returns:
            list: List of numpy arrays if to_numpy=True, else GPU tensor
        """
        # Load frames to GPU (this also sets self.total_frames)
        frames_gpu_tensor = self.load_all_frames_gpu(normalize=False, target_device='cuda:0')

        if not to_numpy:
            # Return GPU tensor (best performance)
            # Squeeze batch dimension: [1, T, C, H, W] -> [T, C, H, W]
            return frames_gpu_tensor.squeeze(0)

        # Convert to numpy (loses GPU acceleration!)
        print("[WARNING] Converting GPU frames to CPU numpy arrays - loses acceleration benefit!")

        # FIX 1: Synchronize GPU before CPU transfer
        torch.cuda.synchronize()

        # frames_gpu_tensor shape: [1, T, C, H, W], range [0, 255] (normalized=False)
        frames_cpu = frames_gpu_tensor[0].cpu()  # [T, C, H, W]

        frames_list = []
        for i in range(frames_cpu.shape[0]):
            # Convert from [C, H, W] to [H, W, C]
            # FIX 2: Make contiguous before numpy
            frame = frames_cpu[i].permute(1, 2, 0).contiguous().numpy().astype('uint8')

            # Convert RGB to BGR if needed
            if color_format == 'BGR':
                frame = frame[:, :, ::-1].copy()  # RGB -> BGR

            frames_list.append(frame)

        return frames_list

    def close(self):
        """Close decoder (legacy compatibility)."""
        # PyNvVideoCodec handles cleanup automatically
        pass

    def __del__(self):
        """Cleanup resources."""
        # PyNvVideoCodec handles cleanup automatically
        pass
