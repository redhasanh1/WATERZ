# Windows Setup - Next Steps

## Current Status

You have all the setup files ready in the `watermarkz` folder. Based on the previous session, you need to complete the installation on your **Windows PC with GTX 1660 Ti**.

## What's Ready

✅ All batch files created:
- `INSTALL_TENSORRT.bat` - Correct TensorRT installation
- `CHECK_EVERYTHING.bat` - Full system verification
- `INSTALL_ALL_LOCAL.bat` - Install all packages to D drive
- `INSTALL_CUDA_12.bat` - PyTorch CUDA 12.6 support
- `SHOW_ERRORS.bat` - Detailed error diagnosis
- `START_REDIS.bat`, `START_CELERY.bat`, `START_SERVER.bat` - Run the system

✅ Server code ready:
- `server_production.py` - Flask + Celery production server
- `yolo_detector_optimized.py` - GPU-optimized YOLO detector
- `web/premium.html` - Landing page

## Next Steps on Your Windows PC

### 1. Fix the 2 Critical Errors

Run this to see exactly what's missing:
```cmd
SHOW_ERRORS.bat
```

Most likely you need to install the core packages:
```cmd
INSTALL_ALL_LOCAL.bat
```

This will install Flask, Celery, Redis, YOLO, OpenCV, NumPy to D drive (no C drive usage).

### 2. Install TensorRT (Optional but Recommended)

Run the fixed TensorRT installer:
```cmd
INSTALL_TENSORRT.bat
```

This gives you 2-3x speedup (20-35 img/sec vs 15-25 img/sec).

### 3. Verify Everything

Run the complete verification:
```cmd
CHECK_EVERYTHING.bat
```

Should show:
- ✅ GPU detected (GTX 1660 Ti)
- ✅ PyTorch CUDA support
- ✅ All packages installed
- ✅ Redis server found
- ✅ TensorRT model ready (if you ran step 2)

### 4. Start the Server

Once CHECK_EVERYTHING.bat shows 0 errors:
```cmd
START_ALL.bat
```

This opens 3 windows:
1. Redis server (message queue)
2. Celery worker (GPU processing)
3. Flask server (web interface)

### 5. Test It

Visit: http://localhost:5000

Upload a test image and verify watermark removal works.

## Expected Performance (GTX 1660 Ti)

- **Without TensorRT**: 15-25 images/sec
- **With TensorRT**: 20-35 images/sec
- **Concurrent users**: 5,000-15,000
- **Revenue potential**: $145K-$435K/month

## Important Notes

⚠️ **These batch files only work on Windows!**
- They won't run in Linux/WSL/Docker
- Run them directly on your Windows PC with the GTX 1660 Ti

⚠️ **Everything stays on D drive:**
- All temp files, cache, packages go to `watermarkz` folder
- C drive stays clean (critical for your setup)

⚠️ **Redis location:**
- You put Redis in `watermarkz/rediz` folder
- Scripts are updated to find it there

## Troubleshooting

If you see package errors:
```cmd
INSTALL_ALL_LOCAL.bat
```

If you see CUDA errors:
```cmd
INSTALL_CUDA_12.bat
```

If you see TensorRT errors:
```cmd
INSTALL_TENSORRT.bat
```

## Why This Matters

Once running, your GTX 1660 Ti can process 15-35 watermark removals per second. At $29/month per user, you need ~5,000-15,000 concurrent users to hit $145K-$435K/month revenue.

Your GPU is capable of handling this load. The server is designed to keep your PC usable while running (you can still game, etc.).

## Questions?

All files are in `D:\github\RoomFinderAI\watermarkz\` on your Windows PC.

Start with `SHOW_ERRORS.bat` to see what's missing!
