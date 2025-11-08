@echo off
REM Build script for DCNv4 TensorRT Plugin on Windows
REM Requires: Visual Studio 2019/2022, CUDA 12.x, TensorRT 10.13.3.9

REM Set CMake path (use Visual Studio's CMake if not in PATH)
set CMAKE_EXE=cmake
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo CMake not in PATH, using Visual Studio's CMake...
    set CMAKE_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)

echo ================================================================================
echo Building DCNv4 TensorRT Plugin
echo ================================================================================
echo.

REM Check if build directory exists
if exist build (
    echo Build directory exists. Cleaning...
    rmdir /s /q build
)

REM Create build directory
echo Creating build directory...
mkdir build
cd build

REM Configure with CMake (Visual Studio 2022)
echo.
echo Configuring with CMake...
%CMAKE_EXE% -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DTENSORRT_DIR=D:/watermarkz/TensorRT-10.13.3.9 ^
      ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo Please check:
    echo   1. Visual Studio 2022 is installed
    echo   2. CUDA 12.x is installed
    echo   3. TensorRT path is correct
    cd ..
    exit /b 1
)

REM Build the plugin
echo.
echo Building plugin...
%CMAKE_EXE% --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    cd ..
    exit /b 1
)

REM Success
echo.
echo ================================================================================
echo Build successful!
echo ================================================================================
echo.
echo Output: build\Release\dcnv4_plugin.dll
echo.
echo To test the plugin:
echo   1. Copy dcnv4_plugin.dll to your Python working directory
echo   2. Run: python test_plugin.py
echo.

cd ..
