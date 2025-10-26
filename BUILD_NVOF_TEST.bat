@echo off
REM Build NVOF Hardware Detection Test for RTX 5070 Blackwell
REM
REM This test will definitively answer: Does RTX 5070 have NVOF hardware?
REM
REM Expected result if hardware exists:
REM   - Status: NV_OF_SUCCESS
REM   - Will print capabilities (grid sizes, resolution limits)
REM
REM Expected result if hardware removed (Blackwell):
REM   - Status: NV_OF_ERR_OF_NOT_AVAILABLE
REM   - Confirms DLSS 4 article claim about AI replacing hardware OF

echo ========================================
echo Building NVOF Hardware Detection Test
echo ========================================
echo.

REM Locate Visual Studio
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"
if not exist "%VS_PATH%\vcvars64.bat" (
    echo ERROR: Visual Studio 2022 not found at expected path
    echo Please update VS_PATH in this script to match your installation
    pause
    exit /b 1
)

REM Setup Visual Studio environment
call "%VS_PATH%\vcvars64.bat"

REM Compile
echo.
echo Compiling test_nvof_hardware.cpp...
echo.

cl /nologo /EHsc test_nvof_hardware.cpp ^
   /I"Optical_Flow_SDK_5.0.7\NvOFInterface" ^
   /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" ^
   /link "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64\cuda.lib"

if errorlevel 1 (
    echo.
    echo ERROR: Compilation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build successful!
echo ========================================
echo.
echo Running test now...
echo.

REM Run the test
test_nvof_hardware.exe

echo.
pause
