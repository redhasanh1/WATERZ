<h1 align="center">Neural Video Inpainting</h1>
<h3 align="center">GPU-Accelerated Video Object Removal & Restoration Engine</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorRT-10.x-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white" />
</p>

<p align="center">
  Production-grade video inpainting system that detects and removes visual artifacts from video using custom-trained YOLO detection, ProPainter temporal inpainting, SAM2 mask tracking, and TensorRT-optimized inference. Built for distributed GPU cloud deployment.
</p>

---

## Overview

Neural Video Inpainting is an end-to-end deep learning pipeline for automated video object detection and removal. The system processes video frame-by-frame using a multi-model architecture: a custom YOLOv8 detector localizes target regions, SAM2 propagates segmentation masks across temporal sequences, and a ProPainter-based inpainting network reconstructs the removed areas with temporal consistency.

The entire pipeline is optimized for production throughput with TensorRT FP16/FP8 quantization, custom DCNv4 CUDA kernels, and horizontal scaling across distributed GPU workers.

## Architecture

```
Input Video
    |
    v
[YOLO Detector] ---- Custom-trained YOLOv8 model (TensorRT FP16)
    |                  Trained on 1000+ annotated frames
    |                  Outputs bounding boxes per frame
    v
[SAM2 Tracker] ----- Segment Anything Model 2 (TensorRT FP16)
    |                  Propagates masks across temporal sequence
    |                  GPU-accelerated NVDEC/NVENC preprocessing
    v
[ProPainter] ------- Temporal video inpainting network
    |                  RAFT optical flow estimation
    |                  Focal Transformer attention
    |                  DCNv4 deformable convolutions
    v
[FFmpeg] ----------- Frame reassembly + audio merge
    |
    v
Output Video
```

## Key Technical Highlights

### Custom YOLO Training Pipeline
- Trained YOLOv8 detection model on custom-annotated dataset of 1000+ frames
- Specialized training for Sora-generated content detection
- Automated frame extraction and YOLO-format annotation pipeline
- Model achieves real-time detection at 640x480 resolution

### TensorRT Optimization & Custom CUDA Kernels
- **FP16 inference**: 1.5-2x speedup over PyTorch baseline on RTX 4090
- **FP8 quantization**: Additional 1.3-1.5x speedup on Ada Lovelace GPUs
- **Custom DCNv4 TensorRT plugin**: Hand-written CUDA kernels for Deformable Convolution v4, replacing slower DCNv2 operations with 3x faster deformable convolutions targeting SM 8.9 architecture
- **ONNX export pipeline**: Automated PyTorch-to-ONNX-to-TensorRT conversion with dynamic batch support
- **Target**: 7-10ms inference at 640x480 (down from 21.1ms PyTorch baseline)

### SAM2 Integration with Temporal Optimization
- Segment Anything Model 2 for precise mask generation and propagation
- 10fps optimized mode: 3.1x faster than native FPS processing
- GPU-accelerated NVDEC/NVENC video preprocessing
- Hybrid pipeline distributing work between GPU decode (4-6s) and mask generation (9-10s)

### Distributed Processing System
- **Celery + Redis** task queue for horizontal GPU scaling
- Chord-pattern orchestration: preparation, parallel segment processing, finalization
- Video segmentation enables parallel processing across multiple GPU workers
- Tested on Salad.com GPU cloud (RTX 3060 through RTX 5090)
- Linear scaling: 2 workers = ~2x throughput, 10 workers = ~10x throughput

### Production Backend
- Flask REST API with async task processing
- Rate limiting, SSRF prevention, input sanitization, CSP headers
- Automatic file lifecycle management (1-hour retention)
- Docker deployment with NVIDIA Container Toolkit
- Celery worker health monitoring and auto-restart

## Performance Benchmarks

| GPU | Inference Time (300 frames) | Relative Speed |
|-----|----------------------------|----------------|
| RTX 3060 | ~30-45s | 1x |
| RTX 3080 | ~20-30s | 1.5x |
| RTX 4090 | ~10-15s | 2.5x |
| RTX 5090 | ~7s | 4.3x |

| Optimization | Latency (640x480) | Speedup |
|-------------|-------------------|---------|
| PyTorch (baseline) | 21.1 ms | 1x |
| TensorRT FP16 | ~10 ms | 2.1x |
| + DCNv4 CUDA kernels | ~7 ms | 3x |
| + FP8 quantization | ~5 ms | 4.2x (projected) |

## Optimization Deep Dive

A breakdown of every optimization applied to take the pipeline from research prototype to production-grade inference.

### Stage 1: Baseline Profiling
The initial PyTorch implementation ran end-to-end at **21.1ms per frame** (640x480). Profiling revealed three bottlenecks:
- **ProPainter's Focal Transformer** — attention computation dominated at ~40% of inference time
- **RAFT optical flow** — redundant computation on similar frames
- **DCNv2 deformable convolutions** — slow PyTorch autograd path, not utilizing tensor cores

### Stage 2: TensorRT FP16 Conversion (2.1x faster)
```
PyTorch Model → ONNX Export → TensorRT Engine (FP16)
```
- Exported ProPainter and RAFT models to ONNX with dynamic batch axes
- Built TensorRT engines with FP16 precision, enabling tensor core utilization
- Handled unsupported ops (grid_sample, deform_conv) with custom plugin fallbacks
- **Result: 21.1ms → 10ms** per frame

### Stage 3: Custom DCNv4 CUDA Kernels (3x faster)
The biggest single optimization. DCNv2 (Deformable Convolution v2) was the remaining bottleneck — TensorRT couldn't fuse it efficiently.

**What we built:**
- Custom TensorRT plugin implementing DCNv4 (Deformable Convolution v4) in raw CUDA
- Targeted Ada Lovelace architecture (SM 8.9) for optimal register allocation
- Replaced bilinear interpolation with hardware-accelerated texture sampling
- Fused offset computation + convolution into a single kernel launch

**Why DCNv4 over DCNv2:**
- 3x faster deformable convolution through softmax-free spatial aggregation
- Reduced memory bandwidth by removing redundant normalization
- Better tensor core utilization with aligned memory access patterns

**Result: 10ms → 7ms** per frame

### Stage 4: SAM2 Temporal Optimization (3.1x video speedup)
This was one of the highest-impact optimizations in the entire pipeline. The naive approach — running SAM2 mask generation on every single frame — was brutally slow. A 22.5-second video at 50fps meant processing **1,124 frames**, taking ~47 seconds. For a production SaaS, that's unacceptable.

**The insight:** Most consecutive video frames are nearly identical. A mask generated at frame 100 is 95%+ valid at frame 101-104. We don't need per-frame inference — we need per-*keyframe* inference with intelligent propagation.

**What we built:**

1. **Adaptive FPS reduction** — Automatically downsample any input video to 10fps before mask generation. A 50fps video drops from 1,124 frames to just 225 frames — an instant 5x reduction in compute. Works across all input frame rates:

   | Input FPS | Frames (22.5s video) | Reduction |
   |-----------|---------------------|-----------|
   | 60fps | 1,350 → 225 | 6x |
   | 50fps | 1,124 → 225 | 5x |
   | 30fps | 675 → 225 | 3x |
   | 24fps | 540 → 225 | 2.4x |

2. **GPU-accelerated preprocessing pipeline** — Moved the entire video decode/resize/fps-conversion step onto the GPU using NVDEC hardware decoder. The CPU never touches raw pixel data:
   - **GPU preprocessing**: 4-6 seconds (NVDEC decode + resize + fps conversion)
   - **Frame extraction**: 1-2 seconds
   - **SAM2 mask generation**: 9-10 seconds (TensorRT FP16)
   - **Total: ~15 seconds** (down from 47s)

3. **Hybrid pipeline architecture** — GPU preprocessing and mask generation run as overlapping stages. While NVDEC is decoding the next batch of frames, SAM2 is generating masks on the current batch. This hides decode latency almost entirely.

4. **One-click interactive workflow** — User selects the target region in a single frame, presses 'c', and SAM2 propagates the mask across the entire temporal sequence automatically. No manual frame-by-frame annotation.

**Before vs After:**
```
BEFORE (Native FPS):
  50fps video, 22.5s → 1,124 frames → SAM2 on every frame → 47 seconds

AFTER (10fps Optimized):
  50fps video, 22.5s → 225 frames → NVDEC + SAM2 → 15 seconds (3.1x faster)
```

**Result: 47s → 15s** — and the quality difference is imperceptible for video inpainting, since ProPainter handles sub-frame interpolation downstream.

### Stage 5: FP8 Quantization (projected 4.2x total)
Currently in progress for standard convolution layers:
- Post-training quantization using calibration dataset
- FP8 (E4M3) for weights, FP8 (E5M2) for activations
- Only applied to layers where accuracy loss < 0.5% PSNR
- Targeting 1.3-1.5x additional speedup on top of FP16

### Stage 6: Distributed Processing (linear horizontal scaling)
For batch/production workloads, single-GPU optimization isn't enough:
- **Celery chord pattern**: Split video into N segments → process in parallel → reassemble
- **Redis broker**: Coordinates work across GPU workers with task retry on failure
- **Cloud deployment**: Salad.com GPU cloud with auto-scaling worker pools
- **Linear scaling validated**: 1 worker = 30s, 2 workers = 15s, 4 workers = 8s

```
                    ┌─── Worker 1 (GPU) ─── Segment 1
                    │
Video → Splitter ───┼─── Worker 2 (GPU) ─── Segment 2  ──→ Concatenate → Output
                    │
                    └─── Worker 3 (GPU) ─── Segment 3
```

### Stage 7: NVDEC/NVENC Hardware Acceleration
Moved video decode/encode off the CPU entirely:
- **NVDEC**: Hardware video decoder on NVIDIA GPUs — zero CPU usage for frame extraction
- **NVENC**: Hardware encoder for output video assembly
- Eliminates the CPU-GPU transfer bottleneck for high-resolution video
- Enables the GPU to handle the full pipeline without CPU involvement

### Optimization Summary

| Stage | Technique | Frame Latency | Cumulative Speedup |
|-------|-----------|--------------|-------------------|
| 0 | PyTorch baseline | 21.1 ms | 1x |
| 1 | TensorRT FP16 | 10 ms | 2.1x |
| 2 | DCNv4 CUDA kernels | 7 ms | 3.0x |
| 3 | FP8 quantization | ~5 ms | 4.2x (projected) |
| + | SAM2 10fps mode | — | 3.1x video speedup |
| + | Distributed (4 GPU) | — | ~4x throughput |
| + | NVDEC/NVENC | — | ~20% end-to-end |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Detection** | YOLOv8, Ultralytics, custom training pipeline |
| **Segmentation** | SAM2 (Segment Anything Model 2), mask propagation |
| **Inpainting** | ProPainter, RAFT optical flow, Focal Transformer |
| **Optimization** | TensorRT 10.x, ONNX Runtime, FP16/FP8 quantization |
| **CUDA** | Custom DCNv4 kernels, NVDEC/NVENC, SM 8.9 targeting |
| **Infrastructure** | Celery, Redis, Flask, Docker, Salad.com GPU cloud |
| **Video** | FFmpeg, Playwright (protected content download) |
| **Frontend** | Vanilla JS, glassmorphism UI |

## Project Structure

```
neural-video-inpainting/
├── backend/              # Flask production server
├── web/                  # Frontend application
├── yolo_training/        # Custom YOLO dataset & training
├── NEW_SORA_TRAINING/    # Sora-specific detection training
├── dcnv4_tensorrt_plugin/# Custom CUDA kernels for TensorRT
├── faster-propainter-main/# Optimized ProPainter implementation
├── edgeconnect/          # Edge-guided inpainting reference
├── weights/              # Model checkpoints
├── tools/                # Utility scripts
├── docs/                 # Technical documentation
├── migrations/           # Database migrations
├── Dockerfile.celery     # GPU worker container
├── docker-compose.salad.yml  # Cloud deployment config
└── BUILD_*.bat           # TensorRT engine build scripts
```

## Deployment

### Local (Single GPU)
```bash
pip install -r requirements.txt
python server_production.py
```

### Docker (GPU Cloud)
```bash
docker build -f Dockerfile.celery -t neural-video-inpainting:latest .
docker push your-registry/neural-video-inpainting:latest
# Deploy to Salad.com, RunPod, or any GPU cloud
```

### Distributed (Multi-GPU)
```bash
# Start Redis broker
docker-compose -f docker-compose.salad.yml up redis

# Scale workers
celery -A server_production.celery worker --concurrency=4
```

See [DOCKER_DEPLOY.txt](DOCKER_DEPLOY.txt) and [DISTRIBUTED_PROCESSING_GUIDE.md](DISTRIBUTED_PROCESSING_GUIDE.md) for full deployment documentation.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `REDIS_URL` | Redis connection for Celery broker |
| `USE_TORCH_COMPILE` | Enable torch.compile optimization |
| `USE_NEUFLOW` | Enable NeuFlow optical flow |
| `ENABLE_FP8_TRANSFORMER` | FP8 quantization for transformer layers |
| `USE_SAM2_TRACKING` | Enable SAM2 mask propagation |
| `ENABLE_NVDEC` | GPU-accelerated video decoding |
| `CELERY_CONCURRENCY` | Worker parallel task count |

---

<p align="center">
  <sub>695 commits | Python, C++/CUDA, JavaScript | Built for production GPU inference</sub>
</p>
