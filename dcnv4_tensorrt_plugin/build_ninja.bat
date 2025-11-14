@echo off
REM Build DCNv4 TensorRT Plugin using Ninja generator (bypasses VS CUDA integration requirement)

echo ================================================================================
echo Building DCNv4 TensorRT Plugin (Ninja generator)
echo ================================================================================
echo.

REM Initialize Visual Studio environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Clean build directory
cd /d D:\watermarkz\dcnv4_tensorrt_plugin
if exist build rmdir /s /q build
mkdir build
cd build

REM Set paths
set CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

echo [*] CMake: %CMAKE_EXE%
echo [*] Ninja: %NINJA_EXE%
echo [*] CUDA:  %CUDA_PATH%
echo.

REM Configure with CMake using Ninja
echo [*] Configuring with CMake (Ninja generator)...
echo.
"%CMAKE_EXE%" -G "Ninja" ^
    -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
    -DTENSORRT_DIR=D:/watermarkz/TensorRT-10.13.3.9 ^
    ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo [*] Building plugin...
echo.
"%CMAKE_EXE%" --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] Build complete!
echo ================================================================================
echo.
echo Output: build\dcnv4_plugin.dll
echo.

cd ..
pause
