"""Test TensorRT NeuFlow v2 optical flow"""
import sys
sys.path.insert(0, r'D:\watermarkz\faster-propainter-main')

import torch
from model.modules.flow_comp_neuflow import NeuFlow_bi

print("[TEST] Initializing TensorRT NeuFlow v2...")
model = NeuFlow_bi(
    model_path=r'D:\watermarkz\faster-propainter-main\models\neuflow_things_fp16.engine',
    device='cuda'
)

print("\n[TEST] Creating test input (batch of 2 frames, 480x640)...")
# Create dummy input: batch=1, time=2, channels=3, height=480, width=640
test_frames = torch.rand(1, 2, 3, 480, 640, device='cuda')

print("[TEST] Computing optical flow...")
with torch.no_grad():
    flows_fwd, flows_bwd = model(test_frames)

print(f"\n[SUCCESS] TensorRT NeuFlow works!")
print(f"  Forward flow shape: {flows_fwd.shape}")
print(f"  Backward flow shape: {flows_bwd.shape}")
print(f"  Flow range: [{flows_fwd.min():.2f}, {flows_fwd.max():.2f}]")
