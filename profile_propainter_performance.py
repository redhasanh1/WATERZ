"""
Profile ProPainter Performance - Identify bottlenecks for optimization
Reports time breakdown for each component:
- Optical flow (NeuFlow TensorRT)
- Feature propagation (BidirectionalPropagation with DCNv4)
- Transformer (SparseWindowAttention)
- Decoder
"""
import os
import sys
import torch
import time
import numpy as np
from contextlib import contextmanager

# Add paths
sys.path.insert(0, 'faster-propainter-main')

# Enable configurations
os.environ['USE_NEUFLOW'] = '1'  # TensorRT NeuFlow
os.environ['ENABLE_FLASH_ATTENTION'] = '1'  # Flash attention

print("=" * 80)
print("PROPAINTER PERFORMANCE PROFILER")
print("=" * 80)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
print()

# Check DCNv4 availability
sys.path.insert(0, r'D:\watermarkz\DCNv4\DCNv4_op')
try:
    from DCNv4 import DCNv4
    dcnv4_available = True
    print("[OK] DCNv4 available (3x faster deformable conv)")
except ImportError:
    dcnv4_available = False
    print("[INFO] DCNv4 not available")

from model.propainter import InpaintGenerator


@contextmanager
def timer(name: str, times_dict: dict):
    """Context manager for timing code blocks"""
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times_dict[name] = elapsed
    print(f"  {name}: {elapsed:.3f}s ({elapsed*1000:.1f}ms)")


def profile_propainter(
    video_frames: int = 30,
    resolution: tuple = (480, 640),  # H, W
    ref_stride: int = 15,
    neighbor_length: int = 10
):
    """Profile ProPainter on synthetic data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    h, w = resolution

    print(f"\n{'='*80}")
    print(f"TEST CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Frames: {video_frames}")
    print(f"  Resolution: {h}x{w}")
    print(f"  Reference stride: {ref_stride}")
    print(f"  Neighbor length: {neighbor_length}")
    print(f"  DCNv4 enabled: {dcnv4_available}")
    print()

    # Initialize model
    print("Loading ProPainter model...")
    model = InpaintGenerator(model_path='weights/ProPainter.pth').to(device)
    model.eval()
    print("[OK] Model loaded")
    print()

    # Create synthetic test data
    print("Creating synthetic test data...")
    with torch.no_grad():
        # Video frames [B, T, C, H, W]
        frames = torch.randn(1, video_frames, 3, h, w, device=device)
        # Binary masks [B, T, 1, H, W] - 20% masked area
        masks = (torch.rand(1, video_frames, 1, h, w, device=device) > 0.8).float()
        # Optical flows [B, T-1, 2, H, W]
        flows_f = torch.randn(1, video_frames-1, 2, h, w, device=device)
        flows_b = torch.randn(1, video_frames-1, 2, h, w, device=device)

        # Masked frames (apply mask to frames)
        masked_frames = frames * (1 - masks)
    print("[OK] Test data created")
    print()

    # Warmup run
    print("Warmup run (compiling kernels)...")
    with torch.no_grad():
        try:
            _ = model(masked_frames, masks, flows_f, flows_b, 'video_completion')
        except Exception as e:
            print(f"[WARNING] Warmup failed: {e}")
    torch.cuda.empty_cache()
    print("[OK] Warmup complete")
    print()

    # Detailed profiling with manual instrumentation
    print(f"{'='*80}")
    print(f"DETAILED COMPONENT PROFILING")
    print(f"{'='*80}")
    print()

    times = {}

    with torch.no_grad():
        # Encoder (expects concatenated masked_frames + 2 masks = 5 channels)
        print("[1/5] Encoder (ResNet-style)")
        with timer("  Encoder", times):
            # ProPainter concatenates: [masked_frames(3), masks_in(1), masks_updated(1)]
            encoder_input = torch.cat([
                masked_frames.view(-1, 3, h, w),
                masks.view(-1, 1, h, w),
                masks.view(-1, 1, h, w)  # Using same mask for simplicity
            ], dim=1)
            enc_feat = model.encoder(encoder_input)
            enc_feat = enc_feat.view(1, video_frames, -1, h//4, w//4)

        # Feature Propagation (BidirectionalPropagation with DCNv4)
        print("\n[2/5] Feature Propagation (DCNv4)")
        with timer("  Feature Propagation (TOTAL)", times):
            # ProPainter uses concatenated masks (masks_in + masks_updated)
            prop_mask_in = torch.cat([
                masks[:, :, :, ::4, ::4],
                masks[:, :, :, ::4, ::4]  # Using same mask for simplicity
            ], dim=2)  # [B, T, 2, H/4, W/4]

            _, _, prop_feat, _ = model.feat_prop_module(
                enc_feat,
                flows_f[:, :, :, ::4, ::4],
                flows_b[:, :, :, ::4, ::4],
                prop_mask_in
            )

        # Transformer (SparseWindowAttention)
        print("\n[3/5] Transformer (Sparse Window Attention)")
        b, t, c, h_feat, w_feat = prop_feat.shape

        # Unfold features for transformer (this is how ProPainter does it)
        prop_feat_flat = prop_feat.view(b*t, c, h_feat, w_feat)
        masks_flat = masks[:, :, :, ::4, ::4].view(b*t, 1, h_feat, w_feat)

        with timer("  Transformer (TOTAL)", times):
            trans_feat = model.transformers(prop_feat_flat, masks_flat)

        trans_feat = trans_feat.view(b, t, c, h_feat, w_feat)

        # Decoder
        print("\n[4/5] Decoder (ResNet-style with upsampling)")
        trans_feat_flat = trans_feat.view(b*t, c, h_feat, w_feat)
        with timer("  Decoder", times):
            pred_img = model.decoder(trans_feat_flat)

        pred_img = pred_img.view(b, t, 3, h, w)

        # Total end-to-end
        print("\n[5/5] End-to-End Inference")
        torch.cuda.empty_cache()
        with timer("  End-to-End (TOTAL)", times):
            _ = model(masked_frames, masks, flows_f, flows_b, 'video_completion')

    # Results summary
    print()
    print(f"{'='*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print()

    # Calculate per-frame times
    encoder_per_frame = times.get("  Encoder", 0) / video_frames * 1000
    prop_per_frame = times.get("  Feature Propagation (TOTAL)", 0) / video_frames * 1000
    trans_per_frame = times.get("  Transformer (TOTAL)", 0) / video_frames * 1000
    decoder_per_frame = times.get("  Decoder", 0) / video_frames * 1000
    total_per_frame = times.get("  End-to-End (TOTAL)", 0) / video_frames * 1000

    print(f"Component Breakdown:")
    print(f"  1. Encoder:              {times.get('  Encoder', 0):.3f}s ({encoder_per_frame:.1f}ms/frame)")
    print(f"  2. Feature Propagation:  {times.get('  Feature Propagation (TOTAL)', 0):.3f}s ({prop_per_frame:.1f}ms/frame) {'[DCNv4]' if dcnv4_available else ''}")
    print(f"  3. Transformer:          {times.get('  Transformer (TOTAL)', 0):.3f}s ({trans_per_frame:.1f}ms/frame)")
    print(f"  4. Decoder:              {times.get('  Decoder', 0):.3f}s ({decoder_per_frame:.1f}ms/frame)")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Total (End-to-End):      {times.get('  End-to-End (TOTAL)', 0):.3f}s ({total_per_frame:.1f}ms/frame)")
    print()

    # Bottleneck analysis
    component_times = {
        "Encoder": times.get("  Encoder", 0),
        "Feature Propagation": times.get("  Feature Propagation (TOTAL)", 0),
        "Transformer": times.get("  Transformer (TOTAL)", 0),
        "Decoder": times.get("  Decoder", 0)
    }

    total_component_time = sum(component_times.values())
    print(f"Bottleneck Analysis (% of total time):")
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
    for name, t in sorted_components:
        pct = (t / total_component_time * 100) if total_component_time > 0 else 0
        print(f"  {name:25s}: {pct:5.1f}%  {'█' * int(pct/2)}")
    print()

    # Optimization recommendations
    print(f"{'='*80}")
    print(f"OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*80}")
    print()

    bottleneck_name, bottleneck_time = sorted_components[0]
    bottleneck_pct = (bottleneck_time / total_component_time * 100)

    print(f"Primary Bottleneck: {bottleneck_name} ({bottleneck_pct:.1f}% of time)")
    print()

    if bottleneck_name == "Feature Propagation":
        if dcnv4_available:
            print("✓ DCNv4 already enabled (3x faster than DCNv2)")
            print("  Next steps:")
            print("  - Option 1: Quantize Conv2d layers in backbone/fuse to FP8 (PyTorch native)")
            print("  - Option 2: Optimize flash attention in DCNv4 (requires custom CUDA)")
            print("  - Option 3: Reduce deform_groups (tradeoff: quality vs speed)")
        else:
            print("⚠ DCNv4 NOT enabled - install DCNv4 for 3x speedup")
            print("  Run: BUILD_DCNV4.bat")

    elif bottleneck_name == "Transformer":
        print("  Optimization options:")
        print("  - Option 1: Use FP8 quantization via Transformer Engine")
        print("  - Option 2: Reduce transformer depths (currently [3, 3, 3])")
        print("  - Option 3: Reduce window size (currently 7)")
        print("  - Option 4: Export transformer to TensorRT with FP8")

    elif bottleneck_name == "Encoder":
        print("  Optimization options:")
        print("  - Option 1: Quantize Conv2d layers to FP8/INT8")
        print("  - Option 2: Use torch.compile on encoder")
        print("  - Option 3: Export encoder to TensorRT")

    elif bottleneck_name == "Decoder":
        print("  Optimization options:")
        print("  - Option 1: Quantize Conv2d layers to FP8/INT8")
        print("  - Option 2: Use torch.compile on decoder")
        print("  - Option 3: Export decoder to TensorRT")

    print()
    print(f"{'='*80}")
    print("PROFILING COMPLETE")
    print(f"{'='*80}")

    return times


if __name__ == "__main__":
    # Profile with realistic settings
    profile_propainter(
        video_frames=30,  # 1 second at 30fps
        resolution=(480, 640),  # Standard video resolution
        ref_stride=15,
        neighbor_length=10
    )
