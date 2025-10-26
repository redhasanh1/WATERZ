@echo off
REM Add TensorRT and (optionally) CUDA to PATH for current session
REM NOTE: Don't use setlocal - we need PATH changes to persist for the calling script

REM Adjust this root if you relocate TensorRT
set "TRT_ROOT=%~dp0TensorRT-10.13.3.9"

if exist "%TRT_ROOT%\lib\nvinfer_10.dll" (
  set "PATH=%TRT_ROOT%\lib;%TRT_ROOT%\bin;%PATH%"
  echo Added TensorRT to PATH: %TRT_ROOT%
) else (
  echo TensorRT not found at %TRT_ROOT%
  echo Edit SETUP_TRT_ENV.bat and set TRT_ROOT accordingly.
  exit /b 1
)

REM Add PyTorch cuDNN DLLs for ONNX Runtime (fixes Conv fallback to CPU)
set "PYTORCH_LIB=%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\torch\lib"
if exist "%PYTORCH_LIB%\cudnn64_9.dll" (
  set "PATH=%PYTORCH_LIB%;%PATH%"
  echo Added PyTorch cuDNN to PATH for ONNX Runtime
) else (
  echo Warning: PyTorch cuDNN DLLs not found at %PYTORCH_LIB%
)

REM If CUDA is installed, add it to PATH (required for cublas/cudnn DLLs)
if defined CUDA_PATH (
  if exist "%CUDA_PATH%\bin" set "PATH=%CUDA_PATH%\bin;%PATH%"
  if exist "%CUDA_PATH%\libnvvp" set "PATH=%CUDA_PATH%\libnvvp;%PATH%"
  echo Using CUDA from CUDA_PATH=%CUDA_PATH%
) else (
  echo CUDA_PATH not set. If trtexec fails on cublas/cudnn DLLs, install CUDA or set CUDA_PATH.
)

REM Quick checks
where trtexec >nul 2>&1 && echo trtexec found || echo trtexec NOT found (ensure TRT bin on PATH)
where nvinfer_plugin_10.dll >nul 2>&1 && echo nvinfer_plugin_10.dll found || echo nvinfer_plugin_10.dll NOT found

echo Environment configured for TensorRT in this terminal.

