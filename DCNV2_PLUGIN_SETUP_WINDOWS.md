DCNv2 TensorRT Plugin — Windows Build Guide

Goal
- Build a DeformableConv2d v2 (DCNv2) TensorRT plugin DLL for use with RFCNet/ProPainter ONNX exports.

Prerequisites
- Visual Studio 2019/2022 (Desktop C++ workload)
- CUDA Toolkit 12.x (matching your driver)
- TensorRT 8.6+ or 10.x (matching CUDA)
- CMake 3.20+

Sources (choose one)
- MMDeploy TRT plugins (includes ModulatedDeformConv2d):
  • https://github.com/open-mmlab/mmdeploy (modules under mmdeploy/csrc/tensorrt)
- Community DCNv2-TensorRT implementation:
  • Repos commonly named DCNv2_TRT / DCN-TRT (match your TRT/CUDA)

Build Steps (MMDeploy example)
1) Get sources (pick a local folder, e.g., D:\src\mmdeploy)
   git clone --recursive https://github.com/open-mmlab/mmdeploy.git

2) Configure CMake (x64, Release)
   -D TensorRT_DIR="C:/TensorRT-10.x.x.x" (path with lib and include)
   -D CUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x"
   -D MMDEPLOY_BUILD_SDK=OFF
   -D MMDEPLOY_BUILD_SDK_PYTHON_API=OFF
   -D MMDEPLOY_TARGET_BACKENDS="trt"

   Example PowerShell (in mmdeploy):
   mkdir build-trt; cd build-trt
   cmake -G "Visual Studio 17 2022" -A x64 ^
     -D CMAKE_BUILD_TYPE=Release ^
     -D TensorRT_DIR="C:/TensorRT-10.7.0.23" ^
     -D CUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" ^
     -D MMDEPLOY_TARGET_BACKENDS=trt ^
     ..

3) Build
   cmake --build . --config Release --target mmdeploy_tensorrt_ops

4) Locate plugin DLL
   - Output typically under build-trt\bin\Release\mmdeploy_tensorrt_ops.dll
   - Rename/copy to a convenient path, e.g., D:\libs\dcnv2_trt_plugin.dll

5) Register the plugin for trtexec
   - Option A: Add the folder to PATH (so the DLL loads automatically)
   - Option B: Pass explicitly via trtexec: --plugins=D:\libs\dcnv2_trt_plugin.dll

Validation
- Run: trtexec --onnx=...\rfcnet.onnx --dryRun --plugins=D:\libs\dcnv2_trt_plugin.dll
- If successful, build the engine with the command in BUILD_RFCNET_ENGINE_FP16.bat

Notes
- Ensure TensorRT, CUDA, and MSVC toolset versions match your local PyTorch/torchvision build to avoid ABI issues.
- For Linux, this produces a .so; on Windows, a .dll.
- If using a different DCNv2 plugin source, map its target name to a stable path and update DCNV2_PLUGIN_DLL accordingly.

