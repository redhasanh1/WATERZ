# WavePaint vs LaMa Inpainting Test

## Quick Start

1. **Get a test image with a Sora watermark** (or any watermark)
   - Place it in the `watermarkz` folder
   - Or have the path ready

2. **Download WavePaint pretrained weights** (REQUIRED)
   - Visit: https://huggingface.co/cloudwalker/WavePaint/tree/main
   - Download: `WavePaint_celebahq.pth` (232 MB)
   - Place in: `watermarkz/wavepaint/WavePaint_celebahq.pth`

   **Direct download link:**
   ```
   https://huggingface.co/cloudwalker/WavePaint/blob/main/WavePaint_celebahq.pth
   ```
   Click "download" button on that page

3. **Run the test**
   ```bash
   TEST_WAVEPAINT.bat
   ```

4. **Compare results**
   - Open `wavepaint_test/results/` folder
   - Compare `03_lama_result.jpg` vs `04_wavepaint_result.jpg`
   - See which one looks better!

## What This Does

```
Test Image (with watermark)
         ↓
    YOLO Detection (your trained model)
         ↓
    Creates Mask
         ↓
    ┌────────┴────────┐
    ↓                 ↓
LaMa Inpaint    WavePaint Inpaint
    ↓                 ↓
Save Result     Save Result
```

## Output Files

- `01_original.jpg` - Your input image
- `02_mask.png` - White = watermark area detected by YOLO
- `03_lama_result.jpg` - LaMa inpainting (current method)
- `04_wavepaint_result.jpg` - WavePaint inpainting (new method)

## Expected Results

**LaMa**: Fast, good for simple watermarks, may leave artifacts

**WavePaint**: Slower, better quality, cleaner edges, fewer artifacts

## If WavePaint is Better

If WavePaint produces better results, you can integrate it into your production server by:

1. Replacing LaMa imports in `server_production.py`
2. Creating a `wavepaint_inpaint.py` wrapper
3. Using WavePaint for all inpainting tasks

## Troubleshooting

**"No watermark detected"**
- Your image may not have a watermark
- Try lowering confidence threshold in the code
- Make sure your YOLO model is loaded correctly

**"WavePaint weights not found"**
- Download the .pth file from the GitHub releases
- Place it in `wavepaint/` folder
- Make sure filename matches exactly

**"CUDA out of memory"**
- WavePaint uses less VRAM than LaMa
- If it fails, you may need to reduce image size
- Try resizing image before testing

## Next Steps

1. **Test on multiple images** - Try different watermark types
2. **Compare quality** - Which looks better to you?
3. **Measure speed** - Which is faster?
4. **Decide** - Keep LaMa or switch to WavePaint?

If WavePaint is better, let me know and I'll integrate it into production!
