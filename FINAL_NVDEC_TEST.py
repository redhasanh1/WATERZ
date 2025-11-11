"""
FINAL NVDEC TEST
Simple decode with RGB output - does it work and is it faster?
"""
import os
os.environ['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin' + os.pathsep + os.environ['PATH']

import cv2
import time
import numpy as np

video_path = 'asser.mp4'

print("=" * 70)
print("FINAL NVDEC TEST - RGB DECODE")
print("=" * 70)
print()

# CPU Baseline
print("[1/2] CPU DECODE")
cap = cv2.VideoCapture(video_path)
cpu_start = time.time()
cpu_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cpu_frames.append(frame)
cpu_time = time.time() - cpu_start
cap.release()

print(f"   Decoded: {len(cpu_frames)} frames")
print(f"   Time: {cpu_time:.3f}s")
print(f"   ms/frame: {(cpu_time / len(cpu_frames)) * 1000:.2f}ms")
print()

# NVDEC Test
print("[2/2] NVDEC RGB DECODE")
try:
    import PyNvVideoCodec as nvc

    # Create demuxer & decoder with RGB output
    demuxer = nvc.CreateDemuxer(filename=video_path)
    codec = demuxer.GetNvCodecId()
    decoder = nvc.CreateDecoder(gpuid=0, codec=codec, outputColorType=nvc.OutputColorType.RGB)

    nvdec_start = time.time()
    nvdec_frames = []

    # Decode all frames
    packet_count = 0
    while packet_count < 1000:  # Max 1000 packets
        try:
            packet = demuxer.Demux()
            if packet is None:
                break

            decoded = decoder.Decode(packet)
            if decoded and len(decoded) > 0:
                for frame in decoded:
                    # DLPack zero-copy conversion to PyTorch, then to numpy (OFFICIAL METHOD!)
                    import torch
                    torch_tensor = torch.from_dlpack(frame)
                    frame_np = torch_tensor.cpu().numpy()

                    # DEBUG: Print first frame info
                    if len(nvdec_frames) == 0:
                        print(f"[DEBUG] First frame shape: {frame_np.shape}, dtype: {frame_np.dtype}")

                    nvdec_frames.append(frame_np)

            packet_count += 1
        except Exception as e:
            print(f"   Decode error: {e}")
            break

    nvdec_time = time.time() - nvdec_start

    print(f"   Decoded: {len(nvdec_frames)} frames")
    print(f"   Time: {nvdec_time:.3f}s")
    print(f"   ms/frame: {(nvdec_time / len(nvdec_frames)) * 1000:.2f}ms")
    print()

    # Results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"   CPU:   {cpu_time:.3f}s ({len(cpu_frames)} frames)")
    print(f"   NVDEC: {nvdec_time:.3f}s ({len(nvdec_frames)} frames)")
    print()

    if len(nvdec_frames) == len(cpu_frames):
        speedup = cpu_time / nvdec_time
        print(f"   SPEEDUP: {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'}!")
        print()

        if speedup > 1.5:
            print("   [OK] NVDEC IS FASTER - USE IT!")
        elif speedup > 1.0:
            print("   [WARN] NVDEC MARGINALLY FASTER - LOW BENEFIT")
        else:
            print("   [SKIP] NVDEC NOT FASTER - STICK WITH CPU!")
    else:
        print(f"   [ERROR] Frame count mismatch - NVDEC FAILED!")

except Exception as e:
    print(f"   [ERROR] NVDEC FAILED: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
