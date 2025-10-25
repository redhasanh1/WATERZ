@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
cd /d E:\watermarkz\mmdeploy
if exist build-trt rmdir /s /q build-trt
mkdir build-trt
cd build-trt
REM Prefer central TensorRT root
call "%~dp0SETUP_TRT_ENV.bat" >nul 2>&1
set "TENSORRT_DIR=%~dp0TensorRT-10.13.3.9"
REM Set CUDA paths for CUDA 12.8
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH_V12_8=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
REM Set cuDNN path
set "CUDNN_DIR=C:\Program Files\NVIDIA\CUDNN\v9.14"
cmake -G "Visual Studio 17 2022" -A x64 ^
  -D CMAKE_BUILD_TYPE=Release ^
  -D TensorRT_DIR=%TENSORRT_DIR% ^
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
