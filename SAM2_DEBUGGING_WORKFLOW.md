# SAM2 Interactive Pipeline - Debugging Workflow

## Problem
Object not being removed from video after running SAM2 interactive tool.

## Debugging Steps

### Step 1: Run SAM2 Interactive in DEBUG Mode

This preserves the masks for analysis:

```batch
RUN_SAM2_INTERACTIVE_DEBUG.bat
```

- Click on the object you want to remove
- Press 'c' to confirm
- Masks will be saved to `temp_sam2_masks/` and preserved

### Step 2: Analyze Mask Quality

Run the visualization tools:

```batch
RUN_VISUALIZE_SAM2_MASKS.bat
```

This will:
1. Print detailed mask statistics (text report)
2. Create `results/sam2_mask_visualization.mp4` (visual debug video)

### Step 3: Interpret Results

The tools will tell you one of these:

#### Scenario A: Good Masks (all frames covered, smooth tracking)
```
✅ Frames with masks: 16/16
✅ Avg coverage: 2.5% per frame
✅ SMOOTH TRACKING - Object moving steadily
```

**Action:** The problem is in the worker code or processing pipeline:
1. Restart Celery worker to load latest code:
   ```batch
   # Stop current worker (Ctrl+C)
   START_CELERY_SAM2_INTERACTIVE.bat
   ```

2. Check worker logs for new mask validation output:
   ```
   [SAM2 INTERACTIVE] Analyzing ALL 16 masks...
   [SAM2 INTERACTIVE] Mask validation:
      - Frames with masks: 16/16
   ```

3. If you don't see the validation output, the worker isn't loading new code

#### Scenario B: Empty/Missing Masks (SAM2 not tracking)
```
❌ CRITICAL: More than 50% of frames have empty masks!
⚠️  Object may be too small or SAM2 under-segmenting
```

**Action:** Fix SAM2 segmentation:
- Click more precisely on object center
- Add multiple positive points on the object
- Try a different object/video
- Adjust SAM2 threshold in code

#### Scenario C: Object Moving Significantly
```
⚠️  SIGNIFICANT MOVEMENT - Fixed crop region may miss object!
   Object travels: 450px width x 320px height
```

**Action:** This is exactly what we fixed! The worker should:
- Analyze ALL masks to calculate union bounding box
- Use the multi-frame crop region

If worker logs don't show "Analyzing ALL 16 masks", restart the worker.

## Expected Worker Output (New Code)

When processing with the updated code, you should see:

```
[SAM2 INTERACTIVE] Analyzing ALL 16 masks to determine crop region...
[SAM2 INTERACTIVE] Mask validation:
   - Frames with masks: 16/16
   - Avg coverage: 2.45% per frame
[SAM2 INTERACTIVE] Detected union bbox (all frames): [245, 380, 512, 720]
   - Object movement: 267px width x 340px height
[SAM2 INTERACTIVE] Crop region: x=225, y=360, w=640, h=480
```

If you don't see this output, the worker is still running OLD code.

## Quick Reference

### Files Created:
- `RUN_SAM2_INTERACTIVE_DEBUG.bat` - Run with mask preservation
- `check_sam2_masks.py` - Text-based mask analysis
- `visualize_sam2_masks.py` - Creates visual debug video
- `RUN_VISUALIZE_SAM2_MASKS.bat` - Runs both analysis tools

### Environment Variables:
- `DEBUG_KEEP_MASKS=1` - Preserve masks after processing

### Output Locations:
- Masks: `temp_sam2_masks/*.png`
- Visualization: `results/sam2_mask_visualization.mp4`
- Final video: `results/test_3se_sam2_removed.mp4`

## Common Issues

### Issue 1: "Masks folder not found"
**Solution:** Run `RUN_SAM2_INTERACTIVE_DEBUG.bat` instead of regular version

### Issue 2: Object not removed despite good masks
**Solution:** Worker running old code - restart with `START_CELERY_SAM2_INTERACTIVE.bat`

### Issue 3: Masks show wrong object
**Solution:** Click more precisely, or add negative points to exclude unwanted regions

### Issue 4: Masks disappear in some frames
**Solution:** SAM2 losing tracking - add more positive points for better consistency
