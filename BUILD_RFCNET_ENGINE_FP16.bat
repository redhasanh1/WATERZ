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

"%TRTEXEC%" --onnx=%ONNX% ^
  --saveEngine=%ENGINE% ^
  --fp16 --memPoolSize=workspace:4096M --useSpinWait ^
  --tacticSources=+CUBLAS,+CUDNN ^
  --minShapes=masked_flows:1x4x2x256x256,masks:1x4x1x256x256 ^
  --optShapes=masked_flows:1x8x2x480x480,masks:1x8x1x480x480 ^
  --maxShapes=masked_flows:1x12x2x640x640,masks:1x12x1x640x640 ^
  --timingCacheFile=%TIMING_CACHE% ^
  --plugins=%DCNV2_PLUGIN_DLL%

if %ERRORLEVEL% NEQ 0 (
  echo Engine build failed.
  exit /b %ERRORLEVEL%
)

echo RFCNet FP16 engine built at %ENGINE%
endlocal
