@echo off
REM Export FastFlowNet (ptlflow) to ONNX using the helper script.
setlocal

set "SCRIPT_DIR=%~dp0"
set "REPO_DIR=%SCRIPT_DIR%.."

pushd "%REPO_DIR%"

if not exist "faster-propainter-main\engines\raft" (
    mkdir "faster-propainter-main\engines\raft"
)

python "faster-propainter-main\tools\export_fastflownet_onnx.py" ^
    --output "faster-propainter-main\engines\raft\fastflownet.onnx" ^
    --min-shape 1x2x3x128x128 ^
    --opt-shape 1x2x3x256x256 ^
    --max-shape 1x2x3x640x640

if errorlevel 1 (
    echo Failed to export FastFlowNet ONNX.
    popd
    endlocal
    exit /b 1
)

echo FastFlowNet ONNX exported to faster-propainter-main\engines\raft\fastflownet.onnx

popd
endlocal
exit /b 0

