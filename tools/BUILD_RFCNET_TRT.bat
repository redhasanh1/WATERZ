@echo off
REM Build TensorRT engine for RFCNet. Requires deformable-conv plugin support.
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_DIR=%SCRIPT_DIR%.."

set "ONNX_PATH=%REPO_DIR%\faster-propainter-main\engines\rfcnet\rfcnet.onnx"
set "ENGINE_DIR=%REPO_DIR%\faster-propainter-main\engines\rfcnet"
set "ENGINE_PATH=%ENGINE_DIR%\rfcnet_fp16.engine"

if not exist "%ONNX_PATH%" (
    echo ONNX file not found: %ONNX_PATH%
    echo Run EXPORT_RFCNET_ONNX.bat first.
    endlocal
    exit /b 1
)

trtexec ^
    --onnx="%ONNX_PATH%" ^
    --saveEngine="%ENGINE_PATH%" ^
    --fp16 ^
    --memPoolSize=workspace:4096 ^
    --minShapes=flows_forward:1x60x2x128x128,flows_backward:1x60x2x128x128,flow_masks:1x61x1x128x128 ^
    --optShapes=flows_forward:1x60x2x256x256,flows_backward:1x60x2x256x256,flow_masks:1x61x1x256x256 ^
    --maxShapes=flows_forward:1x60x2x384x384,flows_backward:1x60x2x384x384,flow_masks:1x61x1x384x384 ^
    --tacticSources=+CUBLAS,+CUDNN ^
    --timingCacheFile="%ENGINE_DIR%\trt_timing_cache"

if errorlevel 1 (
    echo TensorRT engine build failed (likely missing deformable convolution plugin).
    endlocal
    exit /b 1
)

echo RFCNet TensorRT engine saved to %ENGINE_PATH%
endlocal
exit /b 0

