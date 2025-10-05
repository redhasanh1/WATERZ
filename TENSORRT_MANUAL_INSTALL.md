# TensorRT Manual Installation for GTX 1660 Ti

## Step 1: Download TensorRT

1. Go to: https://developer.nvidia.com/tensorrt
2. Click **"Download Now"** or **"Get Started"**
3. You'll need to **create a free NVIDIA Developer account** (or login)
4. Select: **TensorRT 8.6 GA** (compatible with CUDA 11.8)
5. Download: **TensorRT 8.6.1.6 for Windows 10 and CUDA 11.x ZIP Package**
   - File size: ~700MB
   - File name: `TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip`

## Step 2: Extract TensorRT

1. Extract the ZIP to a temporary folder (like `C:\Temp\TensorRT`)
2. You'll see folders: `bin`, `lib`, `include`, `python`, etc.

## Step 3: Install Python Wheel

1. Open the extracted folder
2. Go to: `TensorRT-8.6.1.6\python\`
3. Find the wheel file for Python 3.11:
   - `tensorrt-8.6.1-cp311-none-win_amd64.whl`
4. Copy the full path to this file

5. Open Command Prompt in your watermarkz folder and run:
```cmd
python -m pip install --target python_packages "C:\Temp\TensorRT\TensorRT-8.6.1.6\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
```

(Replace the path with wherever you extracted it)

## Step 4: Copy DLL Files

TensorRT needs DLL files to be accessible:

```cmd
copy "C:\Temp\TensorRT\TensorRT-8.6.1.6\lib\*.dll" python_packages\
```

This copies all TensorRT DLLs to your python_packages folder.

## Step 5: Test TensorRT

Run this to verify it's installed:

```cmd
python -c "import sys; sys.path.insert(0, 'python_packages'); import tensorrt as trt; print('TensorRT version:', trt.__version__)"
```

Should show: `TensorRT version: 8.6.1`

## Step 6: Export YOLO to TensorRT

Now run the export:

```cmd
python -c "import sys; sys.path.insert(0, 'python_packages'); from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='engine', device=0, half=True)"
```

This takes 2-5 minutes. It will create `yolov8n.engine` (~12MB).

## Step 7: Verify

Check if the file exists:

```cmd
dir yolov8n.engine
```

If you see the file, you're done! 🚀

## Troubleshooting

**If you get "DLL not found" errors:**
- Make sure you copied ALL .dll files from `TensorRT\lib\` to `python_packages\`
- You may need to copy them to the same folder as your Python executable too

**If download link doesn't work:**
- The direct link changes. You MUST login to NVIDIA Developer account
- Alternative: https://developer.nvidia.com/nvidia-tensorrt-8x-download

**If Python version doesn't match:**
- You're using Python 3.11 (confirmed from earlier)
- Make sure you download the `cp311` wheel file
- If you have Python 3.10, download `cp310` wheel instead

## Expected Result

After successful installation:
- Speed: 20-35 images/sec (vs 15-25 without TensorRT)
- Revenue potential: +$50K-$100K/month at capacity
- File created: `yolov8n.engine` (~12MB)

Good luck! 💪
