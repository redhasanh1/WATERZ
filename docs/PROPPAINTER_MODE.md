# ProPainter-Only Mode (October 2025)

This setup removes the WavePaint/LaMa inpainting pipeline and routes all GPU work
through ProPainter, driven by YOLOv8 detections.

## What changed
- `server_production.py`
  - Loads only the YOLO detector.
  - Builds per-frame masks in `/temp/propainter_masks/…` and calls
    `ProPainter/inference_propainter.py`.
  - Merges audio back onto the rendered video and serves the final file from `results/`.
  - Rejects still-image requests (`/api/remove-watermark`) with a 400 error.
- `requirements.gpu.txt`
  - Trimmed to ProPainter + YOLO dependencies only (PowerPaint/LaMa/IOPaint removed).
- `SALAD_GPU_WORKER.md`
  - Updated to describe the ProPainter worker flow.

## Testing checklist
1. `docker build -f Dockerfile.gpu -t watermarkz-gpu .`
2. `docker run --rm --gpus all -e REDIS_URL=redis://host.docker.internal:6379/0 watermarkz-gpu`
3. Submit a video through the UI; confirm `results/<name>_propainter.mp4` is created.
4. Watch the Salad logs for “Launching ProPainter” after deployment.

## Rolling back
If you need the old WavePaint/LaMa pipeline again:
1. Restore the affected files from git:  
   `git checkout -- server_production.py requirements.gpu.txt SALAD_GPU_WORKER.md`
2. Rebuild the Docker images (`Dockerfile` and/or `Dockerfile.gpu`) and redeploy.

Keep this file alongside the change so you can track what to undo later.
