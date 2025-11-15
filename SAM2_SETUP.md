# SAM2 Watermark Removal - Parallel Production System

This is a **separate SAM2 system** that runs in parallel with your existing YOLO production setup.

## Files Created

### 1. `server_production2.py`
- Flask server (port 5001) with SAM2 Celery task
- Handles SAM2-based watermark removal with pre-generated masks
- **Does NOT interfere with existing `server_production.py`**

### 2. `START_CELERY_SAM2_LOCAL.bat`
- Celery worker for SAM2 tasks
- Optimized for ProPainter with NeuFlow + FP8 + DCNv4
- **Separate from existing `START_CELERY_TRT.bat`**

### 3. `process_stock2_sam2.py`
- Simple test script to process stock2.mp4
- Uses existing masks from `temp_sam2_masks`

## How It Works

### Workflow:
```
1. Generate SAM2 masks (already done - temp_sam2_masks/)
   └─ 1124 PNG files: 00000.png, 00001.png, ..., 01123.png

2. Start SAM2 Celery worker
   └─ START_CELERY_SAM2_LOCAL.bat

3. Run test script
   └─ python process_stock2_sam2.py

4. Output saved to:
   └─ D:\watermarkz\results\stock2_sam2_removed.mp4
```

## Quick Start

### Step 1: Start the SAM2 Celery Worker
```batch
START_CELERY_SAM2_LOCAL.bat
```

This will:
- Load Redis from `redis_url.txt`
- Enable SAM2 interactive mode
- Configure ProPainter optimizations
- Start 4 parallel workers

### Step 2: Run the Test
```batch
python process_stock2_sam2.py
```

This will:
- Submit SAM2 task to Celery
- Poll for completion
- Show progress updates
- Display output path when complete

### Step 3: Check Output
```
D:\watermarkz\results\stock2_sam2_removed.mp4
```

## API Endpoints

### POST `/api/process_sam2`
Submit SAM2 watermark removal task

**Request:**
```json
{
    "video_path": "D:\\watermarkz\\videostotrain\\stock2.mp4",
    "masks_folder": "D:\\watermarkz\\temp_sam2_masks",
    "video_id": "stock2"
}
```

**Response:**
```json
{
    "task_id": "abc-123-def-456",
    "status": "queued"
}
```

### GET `/api/task/<task_id>`
Check task status

**Response (in progress):**
```json
{
    "state": "PROCESSING",
    "current": 40,
    "status": "Running ProPainter"
}
```

**Response (complete):**
```json
{
    "state": "SUCCESS",
    "result": {
        "status": "success",
        "output_path": "D:\\watermarkz\\results\\stock2_sam2_removed.mp4",
        "total_frames": 1124,
        "width": 1920,
        "height": 1080,
        "fps": 50.0
    }
}
```

## Environment Variables

### SAM2 Configuration
```batch
SAM2_POSITION_TOLERANCE=50      # Allow 50px watermark movement
SAM2_MIN_SEGMENT_LENGTH=3       # Minimum 3 frames per segment
SAM2_MAX_SEGMENTS=80            # Cap at 80 segments
SAM2_STATIONARY_THRESHOLD=10    # <10px movement = stationary
```

### Optimizations
```batch
USE_NEUFLOW=1                   # NeuFlow v2 optical flow (10-70x faster)
FORCE_TRT_RFCNET=1              # RFCNet TensorRT (1.6-2.3x speedup)
ENABLE_DCNV4_RFCNET=1           # DCNv4 plugin (3x speedup)
ENABLE_FP8_TRANSFORMER=1        # FP8 transformer (1.3-1.5x speedup)
ENABLE_FP8_ENCODER=1            # FP8 encoder (1.3-1.5x speedup)
ENABLE_FP8_DECODER=1            # FP8 decoder (1.3-1.5x speedup)
ENABLE_FP8_RFCNET=1             # FP8 RFCNet (1.3-1.5x speedup)
SEGMENT_WORKERS=4               # 4 parallel ProPainter workers
```

## Processing Pipeline

### 1. Mask Analysis
- Loads all SAM2 masks from `temp_sam2_masks`
- Calculates union bounding box across all frames
- Determines watermark movement range

### 2. Cropping
- Crops frames to watermark region (20% padding)
- Crops masks to same region
- Reduces ProPainter processing area

### 3. Segment Detection
- Analyzes mask positions across frames
- Groups frames into motion-based segments
- Creates filler segments for uncovered frames

### 4. Adaptive Processing
- **Stationary watermarks** (<10px movement):
  - neighbor_length=4, subvideo_length=20 (ULTRA-FAST)
- **Moving watermarks** (>10px movement):
  - neighbor_length=6, subvideo_length=40 (BALANCED)

### 5. ProPainter Execution
- Single segment: Process entire video
- Multiple segments: Process each with individual crop
- Clear GPU memory between segments

### 6. Merging
- Paste cleaned segments back into cropped frames
- Merge cropped region into original full frames
- Encode with FFmpeg + preserve audio

## Expected Performance

With RTX 4090 + all optimizations:
- **Optical Flow (NeuFlow v2)**: 10-70x faster than RAFT
- **RFCNet (TensorRT + DCNv4)**: 1.6-2.3x speedup
- **Transformer (FP8)**: 1.3-1.5x speedup
- **Encoder/Decoder (FP8)**: 1.3-1.5x speedup

### Typical Processing Time:
- 1124 frames (stock2.mp4 @ 50fps = 22.5 seconds)
- Estimated: ~3-5 minutes total processing time

## Troubleshooting

### Celery worker won't start
```
Error: Could not connect to Redis
```
**Fix:** Check `redis_url.txt` exists and Redis is running

### Task stays in PENDING
```
State: PENDING, Status: Pending...
```
**Fix:** Celery worker not running, start with `START_CELERY_SAM2_LOCAL.bat`

### API connection error
```
Could not connect to API at http://localhost:5001
```
**Fix:** Flask server not running:
```bash
python server_production2.py
```

### ProPainter assets missing
```
RuntimeError: ProPainter assets missing
```
**Fix:** Download model weights:
```bash
cd faster-propainter-main
python checkpoints/download_ckpts.py
```

## Production vs SAM2 Systems

| Feature | Production (YOLO) | SAM2 (This System) |
|---------|-------------------|-------------------|
| Server | `server_production.py` | `server_production2.py` |
| Port | 5000 | 5001 |
| Celery Batch | `START_CELERY_TRT.bat` | `START_CELERY_SAM2_LOCAL.bat` |
| Detection | YOLO TensorRT | SAM2 masks (pre-generated) |
| Tracking | Optional SAM2 | Always uses SAM2 masks |
| Task Name | `watermark.process_*` | `watermark.process_sam2_interactive` |

**Both systems can run simultaneously without conflicts!**

## Next Steps

1. ✅ Generate SAM2 masks with WSL2:
   ```bash
   .\RUN_SAM2_WSL2_GUI.ps1 "D:\watermarkz\videostotrain\stock2.mp4"
   ```

2. ✅ Start SAM2 Celery worker:
   ```batch
   START_CELERY_SAM2_LOCAL.bat
   ```

3. ✅ Process video:
   ```bash
   python process_stock2_sam2.py
   ```

4. ✅ Check output:
   ```
   D:\watermarkz\results\stock2_sam2_removed.mp4
   ```

## Notes

- Masks are **NOT stored in Redis** (too large)
- Masks are read from local folder (`temp_sam2_masks`)
- Each video needs fresh SAM2 mask generation
- Segments are auto-detected from mask positions
- No YOLO detection in SAM2 mode (masks are ground truth)

---

**Questions?** Check the SAM2 task logs in Celery worker console.
