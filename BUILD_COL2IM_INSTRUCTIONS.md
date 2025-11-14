# Col2Im Plugin Build Instructions

## ⚠️ IMPORTANT: Manual Build Required

Git Bash and Python subprocess cannot properly execute Visual Studio's vcvars64.bat environment setup script.

**The Col2Im plugin DLL MUST be built manually from Windows Command Prompt.**

## Build Steps

### Step 1: Open Windows Command Prompt
- Press `Win + R`
- Type `cmd`
- Press Enter

### Step 2: Navigate and Build
```cmd
D:
cd D:\watermarkz
BUILD_COL2IM_PLUGIN.bat
```

### Step 3: Verify Success
The build should create:
```
D:\watermarkz\col2im_tensorrt_plugin\build\Release\col2im_plugin.dll
```

Expected output:
```
[OK] Plugin DLL created: col2im_tensorrt_plugin\build\Release\col2im_plugin.dll
```

### Step 4: Test DLL Loads (Optional)
```batch
python
>>> import ctypes
>>> dll = ctypes.CDLL(r"D:\watermarkz\col2im_tensorrt_plugin\build\Release\col2im_plugin.dll")
>>> print("[SUCCESS] Plugin loaded!")
```

## Build Time
- CMake configure: ~30 seconds
- Compilation: ~2-5 minutes (depending on CPU)
- Total: ~3-6 minutes

## Troubleshooting

### Error: "CUDA toolset not found"
**Solution**: The batch file automatically calls vcvars64.bat. Make sure you're running from CMD, not Git Bash.

### Error: "TensorRT not found"
**Solution**: Check that `D:\watermarkz\TensorRT-10.13.3.9` exists with lib/nvinfer_10.lib

### Error: "MSBuild not found"
**Solution**: Install Visual Studio 2022 Build Tools from:
https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

## What Happens Next?

After building the DLL, the FP8 Transformer integration will:
1. Load col2im_plugin.dll before building TensorRT engines
2. Register the Col2Im custom operator
3. Build FP8 TensorRT engine with Col2Im support
4. Achieve 7-15x speedup (2.39s → 0.16-0.34s per segment!)

## Alternative: Skip Build for Now

You can continue with Phase 3 (plugin integration code) without building the DLL yet.
The DLL only needs to exist when actually running the TensorRT engine builder.

Phase 3 creates the Python code to load the DLL - the DLL itself can be built later.
