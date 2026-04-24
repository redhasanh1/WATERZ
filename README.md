<h1 align="center">WATERZ</h1>
<h3 align="center">GPU-Accelerated Neural Video Inpainting Engine</h3>

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

WATERZ is an end-to-end deep learning pipeline for automated video object detection and removal. The system processes video frame-by-frame using a multi-model architecture: a custom YOLOv8 detector localizes target regions, SAM2 propagates segmentation masks across temporal sequences, and a ProPainter-based inpainting network reconstructs the removed areas with temporal consistency.

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
WATERZ/
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
docker build -f Dockerfile.celery -t waterz:latest .
docker push your-registry/waterz:latest
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
