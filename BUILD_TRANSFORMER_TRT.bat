@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo BUILD TENSORRT FP16 TRANSFORMER ENGINE
echo ============================================================
echo.
echo Building from: engines/transformer/transformer_dynamic.onnx
echo Output: engines/transformer/transformer_fp16.engine
echo.
echo This will take 5-15 minutes (kernel auto-tuning)
echo ============================================================
echo.

REM Setup TensorRT environment
call SETUP_TRT_ENV.bat

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

REM Build TensorRT engine (no Col2Im plugin needed - fold/unfold removed!)
"%PYTHON_PATH%" build_transformer_trt_engine.py --onnx engines/transformer/transformer_dynamic.onnx --engine engines/transformer/transformer_fp16.engine --fp16 --workspace 8

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo [SUCCESS] TensorRT Engine Build Complete!
    echo ============================================================
    echo Output: engines/transformer/transformer_fp16.engine
    echo.
    echo Next: Test with python test_transformer_trt.py
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo [ERROR] Build failed!
    echo ============================================================
)

pause
