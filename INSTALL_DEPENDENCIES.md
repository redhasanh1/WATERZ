# Install C++ Dependencies for SAM2 TensorRT

You have Visual Studio 2022 and CMake installed. Now you need these libraries:

## Quick Install with vcpkg (Recommended)

### 1. Install vcpkg (5 min)

Open PowerShell as Administrator:

```powershell
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

### 2. Install Dependencies (30-60 min)

```powershell
cd C:\vcpkg
.\vcpkg install opencv[cuda]:x64-windows
.\vcpkg install boost-system:x64-windows
.\vcpkg install boost-filesystem:x64-windows
.\vcpkg install nlohmann-json:x64-windows
```

**This will take 30-60 minutes** as it compiles OpenCV with CUDA support.

### 3. Build SAM2 TensorRT Library

```bash
cd D:\watermarkz\sam2_trt_inference

cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release
```

Output: `build\Release\trt_sam2_infer.dll`

---

## Alternative: Manual Install (If you already have these)

If you already have OpenCV, Boost, etc. installed elsewhere:

1. Edit `D:\watermarkz\sam2_trt_inference\CMakeLists.txt`
2. Update these paths to point to your installations:
   ```cmake
   set(OpenCV_DIR "path/to/opencv/build")
   set(BOOST_ROOT "path/to/boost")
   ```

---

## Test if Dependencies Exist

Run this in PowerShell to check:

```powershell
# Check for OpenCV
ls "C:\opencv\build\x64\vc16\lib\opencv_world*.lib" -ErrorAction SilentlyContinue

# Check for vcpkg installs
ls "C:\vcpkg\installed\x64-windows\lib" -ErrorAction SilentlyContinue
```

---

## Quick Decision

**Do you want me to:**

A. Guide you through vcpkg setup (30-60 min total)
B. Implement Option B (Python-only TensorRT, no C++ needed, 30 min)
C. Implement Option C (Hybrid: TRT encoder + PyTorch decoder, 15 min)

Options B and C give you 3-5x speedup **right now** without waiting for dependency builds.

**Your call!**
