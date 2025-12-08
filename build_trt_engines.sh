#!/bin/bash
# ============================================================
# TensorRT Engine Builder for ProPainter (Linux/Docker)
# Builds NeuFlow + RFCNet engines for optimal performance
# ============================================================
# Usage: ./build_trt_engines.sh
# Must run on target GPU (engines are GPU-architecture specific)
# ============================================================
set -e

echo "=============================================="
echo "TensorRT Engine Builder for ProPainter"
echo "=============================================="

cd /app

# ============================================
# 1. BUILD NEUFLOW TRT ENGINE
# ============================================
echo ""
echo "[1/2] Building NeuFlow TensorRT FP16 Engine..."
echo "Input:  faster-propainter-main/models/neuflow_things.onnx"
echo "Output: faster-propainter-main/models/neuflow_things_fp16.engine"

NEUFLOW_ONNX="faster-propainter-main/models/neuflow_things.onnx"
NEUFLOW_ENGINE="faster-propainter-main/models/neuflow_things_fp16.engine"

if [ ! -f "$NEUFLOW_ONNX" ]; then
    echo "[ERROR] NeuFlow ONNX not found: $NEUFLOW_ONNX"
    exit 1
fi

# Remove existing engine
rm -f "$NEUFLOW_ENGINE"

# Build with TF32 disabled for consistency
export NVIDIA_TF32_OVERRIDE=0

trtexec \
  --onnx="$NEUFLOW_ONNX" \
  --saveEngine="$NEUFLOW_ENGINE" \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --tacticSources=+CUDNN,+CUBLAS,+EDGE_MASK_CONVOLUTIONS \
  --builderOptimizationLevel=5

if [ -f "$NEUFLOW_ENGINE" ]; then
    echo "[OK] NeuFlow engine built: $(ls -lh $NEUFLOW_ENGINE | awk '{print $5}')"
else
    echo "[ERROR] NeuFlow engine build failed!"
    exit 1
fi

# ============================================
# 2. BUILD RFCNET TRT ENGINE (requires DCNv2 plugin)
# ============================================
echo ""
echo "[2/2] Building RFCNet TensorRT FP16 Engine..."
echo "Input:  faster-propainter-main/engines/rfcnet/rfcnet.onnx"
echo "Output: faster-propainter-main/engines/rfcnet/rfcnet_fp16.engine"

RFCNET_ONNX="faster-propainter-main/engines/rfcnet/rfcnet.onnx"
RFCNET_ENGINE="faster-propainter-main/engines/rfcnet/rfcnet_fp16.engine"
RFCNET_CACHE="faster-propainter-main/engines/rfcnet/trt_timing_cache"

# DCNv2 plugin path (Linux .so)
DCNV2_PLUGIN="/app/mmdeploy_tensorrt_ops.so"

if [ ! -f "$RFCNET_ONNX" ]; then
    echo "[ERROR] RFCNet ONNX not found: $RFCNET_ONNX"
    exit 1
fi

if [ ! -f "$DCNV2_PLUGIN" ]; then
    echo "[WARN] DCNv2 plugin not found: $DCNV2_PLUGIN"
    echo "[WARN] RFCNet TRT will be skipped - using PyTorch fallback"
    echo "[WARN] Run build_dcnv2_plugin.sh first to enable RFCNet TRT"
else
    rm -f "$RFCNET_ENGINE"

    trtexec \
      --onnx="$RFCNET_ONNX" \
      --saveEngine="$RFCNET_ENGINE" \
      --fp16 \
      --memPoolSize=workspace:4096 \
      --useSpinWait \
      --tacticSources=+CUBLAS \
      --timingCacheFile="$RFCNET_CACHE" \
      --plugins="$DCNV2_PLUGIN" \
      --minShapes=masked_flows:1x1x2x128x128,masks:1x1x1x128x128 \
      --optShapes=masked_flows:1x70x2x192x192,masks:1x70x1x192x192 \
      --maxShapes=masked_flows:1x80x2x256x256,masks:1x80x1x256x256

    if [ -f "$RFCNET_ENGINE" ]; then
        echo "[OK] RFCNet engine built: $(ls -lh $RFCNET_ENGINE | awk '{print $5}')"
    else
        echo "[ERROR] RFCNet engine build failed!"
        exit 1
    fi
fi

echo ""
echo "=============================================="
echo "BUILD COMPLETE!"
echo "=============================================="
echo ""
echo "Engines created:"
ls -lh faster-propainter-main/models/*.engine 2>/dev/null || echo "  (no NeuFlow engine)"
ls -lh faster-propainter-main/engines/rfcnet/*.engine 2>/dev/null || echo "  (no RFCNet engine)"
echo ""
echo "Enable in environment:"
echo "  USE_NEUFLOW=1"
echo "  FORCE_TRT_RFCNET=1"
