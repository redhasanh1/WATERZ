@echo off
REM Build RFCNet FP16 TensorRT engine (requires DCNv2 TRT plugin)
setlocal

call "%~dp0SETUP_TRT_ENV.bat"

set "TRTEXEC=%~dp0TensorRT-10.7.0.23\bin\trtexec.exe"
set ONNX=faster-propainter-main\engines\rfcnet\rfcnet.onnx
set ENGINE=faster-propainter-main\engines\rfcnet\rfcnet_fp16.engine
set TIMING_CACHE=faster-propainter-main\engines\rfcnet\trt_timing_cache

REM Path to DCNv2 TensorRT plugin DLL (set before running or edit below)
if "%DCNV2_PLUGIN_DLL%"=="" (
  echo DCNV2_PLUGIN_DLL not set. Set it to your dcnv2_trt_plugin.dll path.
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
