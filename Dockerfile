# Production Docker Image for Watermark Removal Pipeline
# Optimized for NVIDIA RTX 5090 with CUDA 12.1 + TensorRT
# Worker-only image (connects to external Redis)

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install TensorRT and core dependencies
RUN pip install --no-cache-dir \
    tensorrt \
    celery[redis] \
    redis \
    opencv-python \
    numpy \
    Pillow \
    ultralytics \
    requests \
    scipy \
    scikit-image \
    ffmpeg-python

# ==============================================================================
# Runtime Stage (Smaller final image)
# ==============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY server_production.py .
COPY yolo_detector.py .

# Copy python_packages directory (custom packages)
COPY python_packages ./python_packages

# Copy ProPainter pipeline
COPY faster-propainter-main ./faster-propainter-main

# Copy TensorRT engines and model weights
COPY runs/detect/new_sora_watermark/weights/best_fp16_batch_rtx5070.engine \
     ./runs/detect/new_sora_watermark/weights/best_fp16_batch_rtx5070.engine
COPY weights ./weights

# Create necessary directories
RUN mkdir -p temp uploads results cache pip_cache

# Environment variables for optimal GPU performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app/python_packages:$PYTHONPATH \
    YOLO_REQUIRE_TENSORRT=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_HOME=/app/cache \
    XDG_CACHE_HOME=/app/cache \
    OPENCV_TEMP_PATH=/app/temp \
    PIP_CACHE_DIR=/app/pip_cache

# Copy and set entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
