@echo off
REM Build RAFT (FastFlowNet) FP16 TensorRT engine
setlocal

call "%~dp0SETUP_TRT_ENV.bat"

set "TRTEXEC=%~dp0TensorRT-10.13.3.9\bin\trtexec.exe"
set "ONNX=faster-propainter-main\engines\raft\fastflownet.onnx"
set "ENGINE=faster-propainter-main\engines\raft\raft_fp16.engine"
set "TIMING=faster-propainter-main\engines\raft\trt_timing_cache"

if not exist "%ONNX%" (
  echo ONNX not found at %ONNX%
  echo Export it with: python faster-propainter-main\tools\export_fastflownet_onnx.py --opset 16
  exit /b 1
)

"%TRTEXEC%" ^
  --onnx=%ONNX% ^
  --saveEngine=%ENGINE% ^
  --fp16 ^
  --memPoolSize=workspace:4096M ^
  --useSpinWait ^
  --tacticSources=+CUBLAS,+CUDNN ^
  --minShapes=images:1x2x3x128x128 ^
  --optShapes=images:1x2x3x256x256 ^
  --maxShapes=images:1x2x3x640x640 ^
  --timingCacheFile=%TIMING%

if errorlevel 1 (
  echo Engine build failed.
  exit /b 1
)

echo Built RAFT FP16 engine at %ENGINE%
endlocal

