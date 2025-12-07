#!/bin/bash
set -e

echo "=============================================="
echo "4090 Worker - TensorRT ProPainter (SAM2 Mode)"
echo "=============================================="

# Build TensorRT engines if they don't exist (first run only)
NEUFLOW_ENGINE="/app/faster-propainter-main/models/neuflow_things_fp16.engine"
RFCNET_ENGINE="/app/faster-propainter-main/engines/rfcnet/rfcnet_fp16.engine"

if [ ! -f "$NEUFLOW_ENGINE" ]; then
    echo "[BUILD] Building NeuFlow v2 TensorRT engine..."
    trtexec --onnx=/app/faster-propainter-main/models/neuflow_things.onnx \
            --saveEngine=$NEUFLOW_ENGINE \
            --fp16 --workspace=4096
    echo "[OK] NeuFlow engine built"
fi

if [ ! -f "$RFCNET_ENGINE" ]; then
    echo "[BUILD] Building RFCNet TensorRT engine..."
    # Load DCNv4 plugin for deformable convolution
    trtexec --onnx=/app/faster-propainter-main/engines/rfcnet/rfcnet.onnx \
            --saveEngine=$RFCNET_ENGINE \
            --fp16 --workspace=4096 \
            --plugins=/app/dcnv4_plugin/libdcnv4_plugin.so
    echo "[OK] RFCNet engine built"
fi

echo "[OK] All TensorRT engines ready"
echo "=============================================="

# Execute the main command (celery worker)
exec "$@"
