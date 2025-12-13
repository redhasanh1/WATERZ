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
    ffmpeg-python \
    flask \
    flask-cors \
    python-dotenv \
    b2sdk \
    pydantic \
    bcrypt \
    tqdm \
    einops

# ==============================================================================
# Runtime Stage (Smaller final image)
# ==============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies + TensorRT tools (for trtexec)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT (includes trtexec)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y tensorrt \
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
COPY server_production2.py .
COPY yolo_detector.py .

# Copy python_packages directory (custom packages)
COPY python_packages ./python_packages

# Copy ProPainter pipeline
COPY faster-propainter-main ./faster-propainter-main

# Copy YOLO model weights (TensorRT engine builds at runtime)
COPY runs/detect/sora_watermark_v2/weights/best.pt \
     ./runs/detect/sora_watermark_v2/weights/best.pt
COPY weights ./weights

# Create necessary directories
RUN mkdir -p temp uploads results cache pip_cache

# Environment variables for optimal GPU performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app/python_packages:$PYTHONPATH \
    YOLO_REQUIRE_TENSORRT=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_HOME=/app/cache \
    XDG_CACHE_HOME=/app/cache \
    OPENCV_TEMP_PATH=/app/temp \
    PIP_CACHE_DIR=/app/pip_cache

# Copy and set entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
