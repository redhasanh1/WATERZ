# 🚨 CRITICAL: Everything on D Drive Only!

## Problem
C drive is almost full - we need **EVERYTHING** on D drive inside the watermarkz folder.

## Solution
All temp files, cache, models, uploads, results, and packages stay in:
```
D:\github\RoomFinderAI\watermarkz\
```

## ✅ What's Protected

### Folder Structure (All on D)
```
watermarkz/
├── temp/                  ← All temporary files (TEMP, TMP, TMPDIR)
├── cache/                 ← PyTorch models, HuggingFace cache
├── pip_cache/             ← Pip downloads
├── uploads/               ← User uploads
├── results/               ← Processed files
├── python_packages/       ← ALL Python libraries
├── redis_data/            ← Redis database
└── web/                   ← Website files
```

### Environment Variables (Forced to D)
```bash
TEMP=D:\github\RoomFinderAI\watermarkz\temp
TMP=D:\github\RoomFinderAI\watermarkz\temp
TMPDIR=D:\github\RoomFinderAI\watermarkz\temp
TORCH_HOME=D:\github\RoomFinderAI\watermarkz\cache
PIP_CACHE_DIR=D:\github\RoomFinderAI\watermarkz\pip_cache
XDG_CACHE_HOME=D:\github\RoomFinderAI\watermarkz\cache
TRANSFORMERS_CACHE=D:\github\RoomFinderAI\watermarkz\cache
HF_HOME=D:\github\RoomFinderAI\watermarkz\cache
OPENCV_TEMP_PATH=D:\github\RoomFinderAI\watermarkz\temp
```

### Python Packages (Installed Locally)
```bash
# ALL packages go to watermarkz/python_packages
pip install --target python_packages <package>
```

### Redis Database (D Drive)
```bash
# redis.conf forces data to D drive
dir D:\github\RoomFinderAI\watermarkz\redis_data
```

## 🚀 Installation (Everything on D)

### Step 1: Run Installation Script
```bash
cd D:\github\RoomFinderAI\watermarkz
INSTALL_ALL_LOCAL.bat
```

This will:
1. ✅ Create all folders on D drive
2. ✅ Set environment variables to D drive
3. ✅ Install ALL Python packages to `python_packages/`
4. ✅ Download models to `cache/`
5. ✅ Configure Redis to use `redis_data/`
6. ✅ Create startup scripts with D drive paths

### Step 2: Verify C Drive is Clean
```bash
VERIFY_D_DRIVE_ONLY.bat
```

Should show:
```
✅✅✅ SUCCESS - C DRIVE IS SAFE! ✅✅✅
```

### Step 3: Start Server (All on D)
```bash
START_ALL.bat
```

Opens 3 windows:
1. **Redis** - Running from `redis_data/`
2. **Celery Worker** - Using `temp/` and `cache/`
3. **Flask Server** - Saving to `uploads/` and `results/`

## 📁 File Locations

| Type | Location | Size Estimate |
|------|----------|---------------|
| Python packages | `python_packages/` | ~5 GB |
| PyTorch models | `cache/` | ~2 GB |
| YOLO model | `cache/` | ~6 MB |
| LaMa model | `cache/` | ~200 MB |
| Pip cache | `pip_cache/` | ~3 GB |
| User uploads | `uploads/` | Dynamic |
| Results | `results/` | Dynamic |
| Redis data | `redis_data/` | ~10 MB |
| Temp files | `temp/` | Dynamic |

**Total:** ~10-15 GB on D drive, **0 bytes on C drive** ✅

## 🔍 How It Works

### server_production.py
```python
# Forces all paths to D drive at startup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, 'temp')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')

# Override environment variables
os.environ['TEMP'] = TEMP_DIR
os.environ['TORCH_HOME'] = CACHE_DIR
```

### Startup Scripts
```batch
REM START_SERVER.bat, START_CELERY.bat, START_REDIS.bat
set TEMP=%~dp0temp
set TORCH_HOME=%~dp0cache
set PIP_CACHE_DIR=%~dp0pip_cache
```

### Package Installation
```batch
REM Install to local folder, cache on D
python -m pip install ^
    --target python_packages ^
    --cache-dir pip_cache ^
    <package>
```

## ⚠️ Common Issues

### "Package not found"
**Problem:** Python can't find packages in `python_packages/`

**Solution:**
```python
import sys
sys.path.insert(0, 'python_packages')
```
Already added to `server_production.py`

### "C drive filling up"
**Problem:** Files still going to C drive

**Solution:**
1. Run `VERIFY_D_DRIVE_ONLY.bat`
2. Check which files are on C
3. Delete C drive cache: `C:\Users\<you>\.cache\torch`
4. Re-run `INSTALL_ALL_LOCAL.bat`

### "Redis won't start"
**Problem:** Redis not using D drive

**Solution:**
```bash
# Start with config file
redis-server.exe redis.conf
```
Config forces: `dir D:\github\RoomFinderAI\watermarkz\redis_data`

### "Models downloading to C"
**Problem:** PyTorch/YOLO downloading to C drive

**Solution:**
```batch
# Set before running
set TORCH_HOME=D:\github\RoomFinderAI\watermarkz\cache
set XDG_CACHE_HOME=D:\github\RoomFinderAI\watermarkz\cache
```
Already in startup scripts!

## 📊 Disk Usage Monitoring

### Check D Drive Usage
```bash
# In watermarkz folder
du -sh temp cache pip_cache python_packages uploads results redis_data
```

### Check C Drive (Should be empty)
```bash
dir "C:\Users\%USERNAME%\AppData\Local\Temp\*watermark*"
dir "C:\Users\%USERNAME%\.cache\torch"
dir "C:\Users\%USERNAME%\.torch"
```

Should show: `File Not Found` ✅

## 🎯 Automated Cleanup

Create scheduled task to clean temp folder:

**Windows Task Scheduler:**
```batch
# Run daily at 3 AM
cd D:\github\RoomFinderAI\watermarkz
del /Q temp\*.*
del /Q uploads\*.*
del /Q results\*.*
```

Keeps only:
- `cache/` (models)
- `python_packages/` (libraries)
- `pip_cache/` (downloads)
- `redis_data/` (database)

## ✅ Verification Checklist

Before running server:

- [ ] Run `INSTALL_ALL_LOCAL.bat` ✓
- [ ] Run `VERIFY_D_DRIVE_ONLY.bat` ✓
- [ ] Check: All folders exist in watermarkz/
- [ ] Check: C drive temp is empty
- [ ] Check: C drive cache is empty
- [ ] Check: Environment variables point to D
- [ ] Run `START_ALL.bat` ✓
- [ ] Test upload at http://localhost:5000
- [ ] Monitor: `temp/` and `uploads/` stay on D
- [ ] Confirm: No files in C:\Users\...\Temp

## 🚀 Production Ready

Once verified:
1. ✅ C drive is safe (0 bytes used)
2. ✅ All files on D drive (10-15 GB)
3. ✅ Server runs without filling C
4. ✅ Models cached on D
5. ✅ Uploads stored on D
6. ✅ Results saved on D
7. ✅ Redis data on D

**You're ready for $1M/month with C drive protected!** 🎉

## 📞 Support

If C drive still fills up:
1. Run `VERIFY_D_DRIVE_ONLY.bat`
2. Check output for warnings
3. Delete any C drive cache folders
4. Re-run `INSTALL_ALL_LOCAL.bat`
5. Restart server with `START_ALL.bat`

Everything should be on D! 🛡️
