@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo Exporting Transformer to ONNX...
call SETUP_TRT_ENV.bat

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

echo Running ONNX export (opset 18 for col2im support)...
"%PYTHON_PATH%" export_transformer_trt.py --out engines/transformer/transformer_dynamic.onnx --opset 18

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] ONNX export complete!
    echo Output: engines/transformer/transformer_dynamic.onnx
) else (
    echo.
    echo [ERROR] ONNX export failed!
)

pause
