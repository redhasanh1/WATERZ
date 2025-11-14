@echo off
REM Build TensorRT engine for RFC Net with DCNv4 Plugin

echo ================================================================================
echo RFC NET TENSORRT ENGINE BUILDER (WITH DCNv4 PLUGIN)
echo ================================================================================
echo.

REM Step 1: Check if ONNX model exists
if not exist "engines\rfcnet\rfcnet_dcnv4.onnx" (
    echo [!] ONNX model not found!
    echo.
    echo Please export ONNX model first:
    echo    python export_rfcnet_dcnv4_onnx.py
    echo.
    pause
    exit /b 1
)

REM Step 2: Check if DCNv4 plugin exists
if not exist "dcnv4_tensorrt_plugin\build\Release\dcnv4_plugin.dll" (
    echo [!] DCNv4 plugin not found!
    echo.
    echo Please build the plugin first:
    echo    cd dcnv4_tensorrt_plugin
    echo    build.bat
    echo.
    pause
    exit /b 1
)

REM Step 3: Build engine with FP16
echo [*] Building TensorRT engine with FP16 precision...
echo.

"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" build_rfcnet_trt_engine.py ^
    --onnx engines/rfcnet/rfcnet_dcnv4.onnx ^
    --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine ^
    --fp16 ^
    --min-shape 1x8x2x256x256 ^
    --opt-shape 1x8x2x640x480 ^
    --max-shape 1x16x2x1280x720 ^
    --workspace 4

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo [SUCCESS] TensorRT engine built!
    echo ================================================================================
    echo.
    echo Engine saved to: engines\rfcnet\rfcnet_dcnv4_fp16.engine
    echo.
    echo Next: Test the engine with test_rfcnet_trt_dcnv4.py
    echo.
) else (
    echo.
    echo ================================================================================
    echo [FAILED] Engine build failed!
    echo ================================================================================
    echo.
    echo Check the error messages above for details.
    echo.
)

pause
