@echo off
REM Add TensorRT and (optionally) CUDA to PATH for current session
setlocal

REM Adjust this root if you relocate TensorRT
set "TRT_ROOT=%~dp0TensorRT-10.7.0.23"

if exist "%TRT_ROOT%\lib\nvinfer_10.dll" (
  set "PATH=%TRT_ROOT%\lib;%TRT_ROOT%\bin;%PATH%"
  echo Added TensorRT to PATH: %TRT_ROOT%
) else (
  echo TensorRT not found at %TRT_ROOT%
  echo Edit SETUP_TRT_ENV.bat and set TRT_ROOT accordingly.
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

endlocal & set "PATH=%PATH%"
echo Environment configured for TensorRT in this terminal.

