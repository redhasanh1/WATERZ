@echo off
REM Export RFCNet ONNX (uses weights/recurrent_flow_completion.pth)
setlocal

set PYTHONPATH=%CD%\faster-propainter-main;%PYTHONPATH%

if not exist faster-propainter-main\engines\rfcnet mkdir faster-propainter-main\engines\rfcnet

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

"%PYTHON_PATH%" faster-propainter-main\tools\export_rfcnet_onnx.py ^
  --weights weights\recurrent_flow_completion.pth ^
  --out faster-propainter-main\engines\rfcnet\rfcnet.onnx ^
  --opset 12

if %ERRORLEVEL% NEQ 0 (
  echo Export failed. Check Python/torchvision install and deform_conv2d availability.
  exit /b %ERRORLEVEL%
 )

echo RFCNet ONNX written to faster-propainter-main\engines\rfcnet\rfcnet.onnx
endlocal
