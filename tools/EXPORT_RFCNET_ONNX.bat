@echo off
REM Export RFCNet (Recurrent Flow Completion) to ONNX.
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_DIR=%SCRIPT_DIR%.."

pushd "%REPO_DIR%"

if not exist "faster-propainter-main\engines\rfcnet" (
    mkdir "faster-propainter-main\engines\rfcnet"
)

python "faster-propainter-main\tools\export_rfcnet_onnx.py" ^
    --weights "web\weights\recurrent_flow_completion.pth" ^
    --output "faster-propainter-main\engines\rfcnet\rfcnet.onnx" ^
    --max-t 60 ^
    --height 256 ^
    --width 256

if errorlevel 1 (
    echo RFCNet ONNX export failed.
    popd
    endlocal
    exit /b 1
)

echo RFCNet ONNX exported to faster-propainter-main\engines\rfcnet\rfcnet.onnx

popd
endlocal
exit /b 0

