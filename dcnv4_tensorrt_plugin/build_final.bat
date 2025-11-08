@echo off
REM Build DCNv4 TensorRT Plugin using Ninja (bypasses VS CUDA integration requirement)

echo ================================================================================
echo Building DCNv4 TensorRT Plugin (Ninja Generator)
echo ================================================================================
echo.

REM Initialize Visual Studio environment (this sets up cl.exe and other tools)
echo [*] Initializing Visual Studio 2022 x64 environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to initialize Visual Studio environment!
    pause
    exit /b 1
)

REM Set CUDA environment
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=%CUDA_PATH%\bin;%PATH%

REM Add Windows SDK tools (rc.exe, mt.exe) to PATH
set WindowsSdkVerBinPath=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64
set PATH=%WindowsSdkVerBinPath%;%PATH%

echo [+] Visual Studio environment initialized
echo [+] CUDA_PATH: %CUDA_PATH%
echo [+] Windows SDK: %WindowsSdkVerBinPath%
echo.

REM Clean and create build directory
cd /d D:\watermarkz\dcnv4_tensorrt_plugin
if exist build rmdir /s /q build
mkdir build
cd build

REM Set tool paths
set CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe

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
    echo.
    pause
    exit /b 1
)

echo.
echo [*] Building plugin with Ninja...
echo.

"%CMAKE_EXE%" --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed!
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] Build complete!
echo ================================================================================
echo.

if exist dcnv4_plugin.dll (
    echo [+] Output: build\dcnv4_plugin.dll
    dir dcnv4_plugin.dll
) else (
    echo [!] WARNING: dcnv4_plugin.dll not found
)

echo.
cd ..
pause
