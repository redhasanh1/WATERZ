@echo off
REM Create cuDNN 8 compatibility DLLs from cuDNN 9

cd D:\watermarkz\TensorRT-10.13.3.9\lib

echo Copying cuDNN 9 DLLs to cuDNN 8 names for backward compatibility...

copy /Y cudnn64_9.dll cudnn64_8.dll
copy /Y cudnn_adv64_9.dll cudnn_adv64_8.dll
copy /Y cudnn_cnn64_9.dll cudnn_cnn64_8.dll
copy /Y cudnn_ops64_9.dll cudnn_ops64_8.dll
copy /Y cudnn_engines_precompiled64_9.dll cudnn_engines_precompiled64_8.dll
copy /Y cudnn_engines_runtime_compiled64_9.dll cudnn_engines_runtime_compiled64_8.dll
copy /Y cudnn_graph64_9.dll cudnn_graph64_8.dll
copy /Y cudnn_heuristic64_9.dll cudnn_heuristic64_8.dll

echo.
echo cuDNN 8 compatibility DLLs created:
dir /B cudnn*_8.dll
