@echo off
echo ================================================================
echo Copying cuDNN DLLs from TensorRT to CUDA bin (fixes Error 127)
echo ================================================================
echo.
echo Source: D:\watermarkz\TensorRT-10.13.3.9\lib\cudnn*.dll
echo Target: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\
echo.
echo NOTE: This requires Administrator privileges!
echo.
pause

copy "D:\watermarkz\TensorRT-10.13.3.9\lib\cudnn*.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\" /Y

echo.
echo ================================================================
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] cuDNN DLLs copied successfully!
    echo.
    echo Copied files:
    dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudnn*.dll" /b
) else (
    echo [ERROR] Copy failed! Make sure you ran this as Administrator.
    echo Right-click this file and select "Run as administrator"
)
echo ================================================================
echo.
pause
