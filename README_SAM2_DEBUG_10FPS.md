# SAM2 Interactive Debug - 10fps GPU Mode

## Overview

Optimized SAM2 mask generation script that processes videos at 10fps using NVIDIA RTX 4090 GPU acceleration for both video preprocessing and mask generation.

## Key Features

### 1. GPU-Accelerated Video Preprocessing (Hybrid Pipeline)
- **NVDEC** hardware decoding (dedicated decoder on RTX 4090)
- **CPU-based FPS filtering** (minimal overhead, avoids CUDA format issues)
- **NVENC** hardware encoding (dedicated encoder on RTX 4090)
- Automatic FPS conversion: 60fps → 10fps (6x fewer frames)
- Decode/encode run on separate hardware units from CUDA cores

### 2. TensorRT SAM2 Inference
- FP16 precision (~20ms per frame on RTX 4090)
- Uses your existing TensorRT engines
- Frame-by-frame prediction with temporal consistency

### 3. Performance Benefits

**For a 22-second video:**
- **60fps input**: 1320 frames → 220 frames @ 10fps = **6x speedup**
- **30fps input**: 660 frames → 220 frames @ 10fps = **3x speedup**

**Total processing time**: ~10-15 seconds
- GPU preprocessing: 3-5 seconds (NVENC/NVDEC)
- Frame extraction: 1-2 seconds
- Mask generation: 4-5 seconds (220 frames × 20ms)

## Usage

### Quick Start
```batch
RUN_SAM2_INTERACTIVE_DEBUG.bat
```

### Manual Usage
```bash
python run_sam2_interactive_debug.py
```

### Workflow
1. **Select Video**: File picker dialog opens
2. **GPU Preprocessing**: Video converted to 10fps (NVDEC/NVENC)
3. **Interactive Selection**: Click on watermark in first frame
   - Left-click: Add positive point (include)
   - Right-click: Add negative point (exclude)
   - 'c': Confirm and start processing
   - 'r': Reset points
   - 'q': Quit
4. **Mask Generation**: TensorRT SAM2 processes all frames
5. **Output**: Masks saved to `temp_sam2_masks/`

## Architecture

```
Input Video (any FPS)
    ↓
[NVDEC] Hardware Decode (GPU dedicated decoder) → System Memory
    ↓
[FPS Filter] Convert to 10fps (CPU, minimal overhead)
    ↓
System Memory → [NVENC] Hardware Encode (GPU dedicated encoder)
    ↓
Temp 10fps Video → Extract Frames
    ↓
[TensorRT SAM2] Mask generation (CUDA cores)
    ↓
Output Masks (temp_sam2_masks/)
```

**Hybrid Pipeline**: Frames pass through system memory for FPS filtering, but GPU hardware accelerates decode/encode. This avoids CUDA pixel format conversion issues while maintaining high performance.

## Hardware Utilization

### RTX 4090 Hardware Units
1. **NVDEC** (Video Decoder): Decodes input video
2. **NVENC** (Video Encoder): Encodes 10fps output
3. **CUDA Cores**: Runs TensorRT SAM2 inference

**Key Benefit**: Video encoding/decoding runs on dedicated hardware, leaving CUDA cores free for SAM2.

## Output

### Masks Directory
```
temp_sam2_masks/
├── 00000.png  (frame 0 mask)
├── 00001.png  (frame 1 mask)
├── 00002.png  (frame 2 mask)
└── ...
```

### Next Steps
After generating masks:

1. **Remove watermark with ProPainter**:
   ```powershell
   .\RUN_SAM2_WSL2_WITH_MASKS.ps1
   ```

2. **Validate mask quality**:
   ```bash
   python check_sam2_masks.py
   ```

## Comparison with Other Methods

| Method | Input FPS | Frames to Process | Speed | Quality |
|--------|-----------|------------------|-------|---------|
| **run_sam2_interactive_debug.py** | 60 → 10 | 220 | ⚡⚡⚡ Fastest | ✓ Good |
| test_sam2_interactive_10fps.py | Any (keyframes) | 1320 (22 keyframes) | ⚡⚡ Fast | ✓ Good |
| test_sam2_interactive_local.py | Native (60) | 1320 | ⚡ Slow | ✓✓ Excellent |
| test_sam2_wsl2.py | Native (60) | 1320 | ⚡ Medium | ✓✓ Excellent |

## Configuration

### Target FPS
Edit `run_sam2_interactive_debug.py` line 38:
```python
TARGET_FPS = 10  # Change to 5, 15, 20, etc.
```

### TensorRT Engines
Paths in `run_sam2_interactive_debug.py` lines 33-34:
```python
ENCODER_ENGINE = r"D:\watermarkz\sam2_trt_inference\engines\sam2_encoder_fp16.engine"
DECODER_ENGINE = r"D:\watermarkz\sam2_trt_inference\engines\sam2_decoder_fp16_dynamic.engine"
```

## Troubleshooting

### NVENC/NVDEC Not Available
If FFmpeg can't use NVENC/NVDEC:
- Check NVIDIA drivers are up to date
- Run: `ffmpeg -encoders | grep nvenc`
- Run: `ffmpeg -decoders | grep cuvid`

### Fallback to CPU
If GPU encoding fails, the script will show an error. You can modify the FFmpeg command in `preprocess_video_gpu()` to use CPU:
```python
# Change from:
"-c:v", "h264_nvenc"
# To:
"-c:v", "libx264"
```

### Memory Issues
If you run out of GPU memory:
- Close other GPU-intensive applications
- Lower TARGET_FPS (e.g., 5 instead of 10)
- Process shorter video clips

## Performance Tips

1. **For 60fps videos**: Use 10fps (6x speedup)
2. **For 30fps videos**: Use 10fps (3x speedup) or 15fps (2x speedup)
3. **For 24fps videos**: Use 8fps (3x speedup) or 12fps (2x speedup)

## Requirements

- NVIDIA RTX GPU (3000/4000 series) with NVENC/NVDEC
- Python 3.12
- TensorRT 10.x
- FFmpeg with CUDA support (included in static-ffmpeg)
- Required engines: sam2_encoder_fp16.engine, sam2_decoder_fp16_dynamic.engine

## Files

- `run_sam2_interactive_debug.py` - Main script
- `RUN_SAM2_INTERACTIVE_DEBUG.bat` - Launcher
- `sam2_trt_predictor.py` - TensorRT predictor (existing)

## Credits

Built on:
- SAM2 by Meta AI (Segment Anything 2)
- TensorRT optimization by TIER IV
- FFmpeg NVENC/NVDEC by NVIDIA
