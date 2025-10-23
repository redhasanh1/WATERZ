@echo off
setlocal
call D:\product\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64
cd /d D:\github\RoomFinderAI\watermarkz\mmdeploy
if exist build-trt rmdir /s /q build-trt
mkdir build-trt
cd build-trt
set "TENSORRT_DIR=D:\github\RoomFinderAI\watermarkz\TensorRT-10.7.0.23"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
cmake -G "Visual Studio 17 2022" -A x64 ^
  -D CMAKE_BUILD_TYPE=Release ^
  -D TensorRT_DIR=%TENSORRT_DIR% ^
  -D CUDA_TOOLKIT_ROOT_DIR=%CUDA_ROOT% ^
  -D MMDEPLOY_TARGET_BACKENDS=trt ^
  -D MMDEPLOY_BUILD_SDK=OFF ^
  -D MMDEPLOY_BUILD_SDK_PYTHON_API=OFF ^
  ..
if errorlevel 1 goto :error
cmake --build . --config Release --target mmdeploy_tensorrt_ops
if errorlevel 1 goto :error
echo Plugin build complete.
endlocal
exit /b 0
:error
echo Plugin build failed with error %errorlevel%.
endlocal
exit /b %errorlevel%
