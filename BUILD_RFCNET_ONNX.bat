@echo off
REM Export RFCNet ONNX (uses weights/recurrent_flow_completion.pth)
setlocal

set PYTHONPATH=%CD%\faster-propainter-main;%PYTHONPATH%

if not exist faster-propainter-main\engines\rfcnet mkdir faster-propainter-main\engines\rfcnet

REM Use explicit Python path to avoid Windows Store stub
set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Export with conservative dynamic shapes for TensorRT (Blackwell GPU compatibility)
REM Min: 1 frame, 128x128 (small crop)
REM Opt: 70 frames, 192x192 (typical segment with 176px watermark)
REM Max: 80 frames, 256x256 (max segment size, large watermark)
"%PYTHON_PATH%" faster-propainter-main\tools\export_rfcnet_onnx.py ^
  --weights weights\recurrent_flow_completion.pth ^
  --out faster-propainter-main\engines\rfcnet\rfcnet.onnx ^
  --opset 12 ^
  --minshape 1x1x2x128x128 ^
  --optshape 1x70x2x192x192 ^
  --maxshape 1x80x2x256x256

if %ERRORLEVEL% NEQ 0 (
  echo Export failed. Check Python/torchvision install and deform_conv2d availability.
  exit /b %ERRORLEVEL%
 )

echo RFCNet ONNX written to faster-propainter-main\engines\rfcnet\rfcnet.onnx
endlocal
