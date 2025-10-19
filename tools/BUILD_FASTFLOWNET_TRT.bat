@echo off
REM Build a TensorRT engine for FastFlowNet from the exported ONNX file.
REM Requires `trtexec` to be available in PATH (part of the TensorRT toolkit).
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_DIR=%SCRIPT_DIR%.."

set "ONNX_PATH=%REPO_DIR%\faster-propainter-main\engines\raft\fastflownet.onnx"
set "ENGINE_DIR=%REPO_DIR%\faster-propainter-main\engines\raft"
set "ENGINE_PATH=%ENGINE_DIR%\raft_fp16.engine"

if not exist "%ENGINE_DIR%" (
    mkdir "%ENGINE_DIR%"
)

if not exist "%ONNX_PATH%" (
    echo ONNX file not found at %ONNX_PATH%
    echo Run EXPORT_FASTFLOWNET_ONNX.bat first.
    endlocal
    exit /b 1
)

trtexec ^
    --onnx="%ONNX_PATH%" ^
    --saveEngine="%ENGINE_PATH%" ^
    --fp16 ^
    --memPoolSize=workspace:4096 ^
    --minShapes=images:1x2x3x128x128 ^
    --optShapes=images:1x2x3x256x256 ^
    --maxShapes=images:1x2x3x640x640 ^
    --tacticSources=+CUBLAS,+CUDNN ^
    --timingCacheFile="%ENGINE_DIR%\trt_timing_cache"

if errorlevel 1 (
    echo TensorRT engine build failed.
    endlocal
    exit /b 1
)

echo FastFlowNet TensorRT engine saved to %ENGINE_PATH%
endlocal
exit /b 0
