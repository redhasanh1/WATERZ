@echo off
cd /d D:\watermarkz\dcnv4_tensorrt_plugin\build

echo Running CMake configuration...
echo.

"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=D:/watermarkz/TensorRT-10.13.3.9 .. 2>&1

echo.
echo CMake exit code: %ERRORLEVEL%
