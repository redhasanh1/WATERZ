#!/bin/bash
# ============================================================
# Build DCNv2 TensorRT Plugin for RFCNet (Linux/Docker)
# This plugin is required for FORCE_TRT_RFCNET=1
# ============================================================
# Usage: ./build_dcnv2_plugin.sh
# Run this BEFORE build_trt_engines.sh
# ============================================================
set -e

PLUGIN_PATH="/app/mmdeploy_tensorrt_ops.so"

if [ -f "$PLUGIN_PATH" ]; then
    echo "[OK] DCNv2 plugin already exists: $PLUGIN_PATH"
    ls -lh "$PLUGIN_PATH"
    exit 0
fi

echo "=============================================="
echo "Building DCNv2 TensorRT Plugin (MMDeploy)"
echo "=============================================="
echo ""
echo "This will take 5-10 minutes..."
echo ""

# Install build dependencies
apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git

# Clone MMDeploy
cd /tmp
rm -rf mmdeploy
git clone --depth 1 https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git submodule update --init --recursive

# Build only the TensorRT plugin
mkdir -p build && cd build
cmake .. \
    -DMMDEPLOY_TARGET_BACKENDS="trt" \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc) mmdeploy_tensorrt_ops

# Copy plugin to /app
if [ -f "lib/libmmdeploy_tensorrt_ops.so" ]; then
    cp lib/libmmdeploy_tensorrt_ops.so "$PLUGIN_PATH"
    echo ""
    echo "[OK] DCNv2 plugin built successfully!"
    ls -lh "$PLUGIN_PATH"
else
    echo "[ERROR] Plugin build failed!"
    exit 1
fi

# Cleanup
cd /
rm -rf /tmp/mmdeploy

echo ""
echo "=============================================="
echo "DCNv2 Plugin Ready!"
echo "=============================================="
echo ""
echo "Now run: ./build_trt_engines.sh"
