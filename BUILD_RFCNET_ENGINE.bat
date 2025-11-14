@echo off
REM Build RFC Net TensorRT Engine with DCNv4 Plugin

REM Add TensorRT and CUDA to PATH
set PATH=D:\watermarkz\TensorRT-10.13.3.9\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM Run Python build script
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" build_rfcnet_trt_engine.py ^
    --onnx engines/rfcnet/rfcnet_dcnv4.onnx ^
    --engine engines/rfcnet/rfcnet_dcnv4_fp16.engine ^
    --fp16 ^
    --verbose

pause
