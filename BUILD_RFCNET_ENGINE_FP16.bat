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

REM NOTE: cuDNN not supported on Blackwell (RTX 50-series), use CUBLAS only
"%TRTEXEC%" --onnx=%ONNX% ^
  --saveEngine=%ENGINE% ^
  --fp16 --memPoolSize=workspace:4096M --useSpinWait ^
  --tacticSources=+CUBLAS ^
  --timingCacheFile=%TIMING_CACHE% ^
  --plugins="%PLUGIN_PATH%"

if %ERRORLEVEL% NEQ 0 (
  echo Engine build failed.
  exit /b %ERRORLEVEL%
)

echo RFCNet FP16 engine built at %ENGINE%
endlocal
