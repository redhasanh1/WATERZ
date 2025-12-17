# ============================================================
# 4090 ProPainter Worker - TensorRT Edition
# SAM2 Interactive Mode (masks from user clicks, NO YOLO)
# ============================================================
# Uses NVIDIA TensorRT container with trtexec built-in
# Builds 3 components:
#   1. DCNv4 TensorRT Plugin (during docker build)
#   2. NeuFlow TRT Engine (on first run)
#   3. RFCNet DCNv4 TRT Engine (on first run)
# ============================================================

FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages + build tools for DCNv4 plugin
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    cmake \
    ninja-build \
    tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================
# BUILD 1: DCNv4 TensorRT Plugin (compiled during docker build)
# ============================================================
COPY dcnv4_tensorrt_plugin/ ./dcnv4_tensorrt_plugin/

# Build DCNv4 plugin - outputs libdcnv4_plugin.so
# Use fresh build_linux dir to avoid Windows CMakeCache conflicts
RUN cd dcnv4_tensorrt_plugin && \
    rm -rf build_linux && mkdir -p build_linux && cd build_linux && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DTENSORRT_DIR=/usr/src/tensorrt \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;89" \
        -GNinja && \
    ninja && \
    cp libdcnv4_plugin.so /app/libdcnv4_plugin.so && \
    echo "[BUILD] DCNv4 TensorRT plugin compiled successfully"

# ============================================================
# Install Python dependencies
# ============================================================
COPY requirements.docker.trt.txt .
RUN pip install --no-cache-dir -r requirements.docker.trt.txt

# ---- Core Python Files ----
COPY server_production.py .
COPY server_production2.py .
COPY segment_detector.py .
COPY crop_utils.py .
COPY yolo_detector.py .
# Note: watermark.py is in faster-propainter-main/ (copied below)

# ---- YOLO Model Weights (for YOLO-mode tasks from celery queue) ----
COPY runs/detect/sora_watermark_v2/weights/best.pt ./runs/detect/sora_watermark_v2/weights/

# ---- ProPainter Pipeline (entire directory) ----
COPY faster-propainter-main/ ./faster-propainter-main/

# ---- ONNX Models for TRT Engine Building ----
# These will be converted to TRT engines on first run
COPY engines/rfcnet/rfcnet_dcnv4.onnx ./engines/rfcnet/

# ---- PRE-BUILT TensorRT Engines (WSL 4090) ----
# Built on WSL2 with RTX 4090 - copy to FINAL location (no runtime copy needed)
# This allows running with --read-only filesystem
COPY engines_trt/neuflow_things_fp16.engine ./faster-propainter-main/models/neuflow_things_fp16.engine
COPY engines_trt/rfcnet/rfcnet_dcnv4_fp16.engine ./engines/rfcnet/rfcnet_dcnv4_fp16.engine

# ---- Symlink for weights (server_production.py looks in /app/weights/) ----
RUN ln -sf /app/faster-propainter-main/weights /app/weights

# ---- Entrypoint script (builds TRT engines on first run) ----
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# ============================================================
# Environment Variables - TENSORRT ENABLED
# ============================================================

# TensorRT ENABLED - NeuFlow + RFCNet (10-70x faster optical flow!)
ENV USE_NEUFLOW=1 \
    FORCE_TRT_RFCNET=1 \
    ENABLE_DCNV4_RFCNET=1 \
    FORCE_TRT_RAFT=0 \
    FORCE_TRT_TRANSFORMER=0 \
    YOLO_REQUIRE_TENSORRT=0

# DCNv4 Plugin path (built above)
ENV DCNV4_PLUGIN_PATH=/app/libdcnv4_plugin.so

# FP8 Optimizations (RTX 40xx SM89)
ENV ENABLE_FP8_TRANSFORMER=1 \
    ENABLE_FP8_ENCODER=1 \
    ENABLE_FP8_DECODER=1 \
    ENABLE_FP8_RFCNET=1

# SAM2 Interactive Mode (NO YOLO - masks from user clicks)
ENV USE_INTERACTIVE_SAM2=1 \
    USE_SAM2_TRACKING=0 \
    SAM2_USE_SEGMENTS=1 \
    SAM2_PARALLEL_SEGMENTS=1 \
    SAM2_SEGMENT_DETECTION_MODE=full \
    SAM2_MASK_DILATION=4

# Segment detection (motion-based)
ENV SEGMENT_USE_MOTION_DETECTION=1 \
    SEGMENT_MOTION_THRESHOLD=8 \
    SEGMENT_MIN_LEN_FULL=3 \
    SEGMENT_MERGE_GAP_FULL=10 \
    MAX_SEGMENT_FRAMES=300 \
    MAX_SEGMENT_PIXELS=400000 \
    MAX_CROP_PIXELS=150000

# Disabled features
ENV ENABLE_SAGE_ATTENTION=0 \
    ENABLE_FLASH_ATTENTION=0 \
    ENABLE_NVDEC=0 \
    USE_TORCH_COMPILE=0 \
    ENABLE_TOKEN_MERGING=0

# Worker config
ENV CELERY_POOL=threads \
    CELERY_CONCURRENCY=4 \
    SEGMENT_WORKERS=2

# Python path
ENV PYTHONPATH=/app:/app/faster-propainter-main

# API Base URL (NO www!)
ENV API_BASE_URL=https://markremoverai.com

# ============================================================
# ENTRYPOINT - Builds TRT engines on first run, then starts worker
# ============================================================
# Build 2 & 3 happen here (NeuFlow + RFCNet TRT engines)
ENTRYPOINT ["/app/docker-entrypoint.sh"]
