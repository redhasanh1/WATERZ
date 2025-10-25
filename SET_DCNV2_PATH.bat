@echo off
REM Set DCNv2 TensorRT plugin DLL path for engine builds
REM This plugin is required for RFCNet and ProPainter (deformable convolution ops)

set "DCNV2_PLUGIN_DLL=%~dp0mmdeploy\build-trt\bin\Release\mmdeploy_tensorrt_ops.dll"

if not exist "%DCNV2_PLUGIN_DLL%" (
    echo ERROR: DCNv2 plugin DLL not found at:
    echo   %DCNV2_PLUGIN_DLL%
    echo.
    echo Build it first with: build_dcnv2_plugin.bat
    exit /b 1
)

echo DCNv2 plugin path set:
echo   DCNV2_PLUGIN_DLL=%DCNV2_PLUGIN_DLL%
