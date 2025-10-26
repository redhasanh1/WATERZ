@echo off
REM Build RFCNet FP16 TensorRT engine (requires DCNv2 TRT plugin)
setlocal

call "%~dp0SETUP_TRT_ENV.bat"
call "%~dp0SET_DCNV2_PATH.bat"
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

set "TRTEXEC=trtexec"
set ONNX=faster-propainter-main\engines\rfcnet\rfcnet.onnx
set ENGINE=faster-propainter-main\engines\rfcnet\rfcnet_fp16.engine
set TIMING_CACHE=faster-propainter-main\engines\rfcnet\trt_timing_cache

if not exist "%ONNX%" (
  echo ERROR: ONNX model not found: %ONNX%
  echo Run BUILD_RFCNET_ONNX.bat first to export the model.
  exit /b 1
)

set "PLUGIN_PATH=%~dp0TensorRT-10.13.3.9\lib\mmdeploy_tensorrt_ops.dll"
if not exist "%PLUGIN_PATH%" (
  echo ERROR: DCNv2 plugin not found at: %PLUGIN_PATH%
  echo Run: copy mmdeploy\build-trt\bin\Release\mmdeploy_tensorrt_ops.dll TensorRT-10.13.3.9\lib\
  exit /b 1
)

REM NOTE: cuDNN NOT supported on Blackwell (RTX 50-series) - TensorRT explicitly rejects it
REM Use CUBLAS-only with conservative dynamic shapes for reliable tactic selection:
REM   Min: 1 frame, 128x128
REM   Opt: 70 frames, 192x192
REM   Max: 80 frames, 256x256
"%TRTEXEC%" --onnx=%ONNX% ^
  --saveEngine=%ENGINE% ^
  --fp16 --memPoolSize=workspace:4096M --useSpinWait ^
  --tacticSources=+CUBLAS ^
  --timingCacheFile=%TIMING_CACHE% ^
  --plugins="%PLUGIN_PATH%" ^
  --minShapes=masked_flows:1x1x2x128x128,masks:1x1x1x128x128 ^
  --optShapes=masked_flows:1x70x2x192x192,masks:1x70x1x192x192 ^
  --maxShapes=masked_flows:1x80x2x256x256,masks:1x80x1x256x256

if %ERRORLEVEL% NEQ 0 (
  echo Engine build failed.
  exit /b %ERRORLEVEL%
)

echo RFCNet FP16 engine built at %ENGINE%
endlocal
