# E2FGVI Setup (Professional Video Inpainting)

E2FGVI requires complex conda environment. Simpler approach:

## Use ProPainter (Already Installed!)

ProPainter IS flow-guided video inpainting - same as E2FGVI approach.
We had memory issues before, but we can fix it with proper settings.

### Solution: Run ProPainter with SAM2 masks

1. Use SAM2 to detect watermark precisely (better than templates)
2. Feed masks to ProPainter
3. ProPainter handles temporal consistency automatically

This gives professional results without new dependencies!

Run: `python setup_sam2_propainter.py`
