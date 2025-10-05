# ProPainter Watermark Removal - Windows Installation

Run these commands in PowerShell on your Windows machine (NOT in container):

## Step 1: Navigate to watermarkz folder
```powershell
cd D:\github\RoomFinderAI\watermarkz
```

## Step 2: Clone ProPainter (if not already cloned)
```powershell
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter
```

## Step 3: Install Python dependencies
```powershell
# If you have NVIDIA GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# If CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements:
pip install -r requirements.txt
```

## Step 4: Create mask for watermark area
You need to create a mask image showing where the watermark is. The easiest way:

1. Extract first frame from video:
```powershell
ffmpeg -i ../sora_with_watermark.mp4 -vf "select=eq(n\,0)" -q:v 3 frame.jpg
```

2. Use paint/photoshop to create a mask:
   - White = area to remove (watermark)
   - Black = area to keep
   - Save as `mask.png`

## Step 5: Run watermark removal
```powershell
python inference_propainter.py --video ../sora_with_watermark.mp4 --mask mask.png --fp16
```

Output will be saved in `results/` folder.

## Alternative: Use IOPaint (easier for beginners)
```powershell
pip install iopaint
iopaint start --model=lama --device=cuda --port=8080
```

Then open http://localhost:8080 in browser and manually mark the watermark area.
