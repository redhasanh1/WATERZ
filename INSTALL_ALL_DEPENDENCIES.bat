@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo SAM2 TensorRT - Complete Dependency Installation
echo ============================================================
echo.
echo This will install:
echo   1. vcpkg (package manager)
echo   2. OpenCV with CUDA
echo   3. Boost libraries
echo   4. nlohmann-json
echo.
echo Total time: ~30-60 minutes
echo ============================================================
echo.
pause

REM Step 1: Clone vcpkg if needed
echo [Step 1/5] Installing vcpkg...
if exist "C:\vcpkg\vcpkg.exe" (
    echo vcpkg already installed!
) else (
    if exist "C:\vcpkg" (
        echo vcpkg directory exists, bootstrapping...
        cd /d C:\vcpkg
        call bootstrap-vcpkg.bat
    ) else (
        echo Cloning vcpkg...
        cd /d C:\
        git clone https://github.com/microsoft/vcpkg.git
        cd vcpkg
        call bootstrap-vcpkg.bat
    )
)

if not exist "C:\vcpkg\vcpkg.exe" (
    echo ERROR: vcpkg installation failed!
    pause
    exit /b 1
)

echo.
echo [OK] vcpkg installed at C:\vcpkg
echo.

REM Step 2: Integrate vcpkg with Visual Studio
echo [Step 2/5] Integrating vcpkg with Visual Studio...
cd /d C:\vcpkg
vcpkg integrate install

echo.
echo [Step 3/5] Installing nlohmann-json (~1 min)...
vcpkg install nlohmann-json:x64-windows

echo.
echo [Step 4/5] Installing Boost (~5-10 min)...
vcpkg install boost-system:x64-windows boost-filesystem:x64-windows

echo.
echo [Step 5/5] Installing OpenCV with CUDA (~20-40 min)...
echo This is the longest step - it compiles OpenCV from source with CUDA support
echo.
vcpkg install opencv[cuda]:x64-windows

if errorlevel 1 (
    echo.
    echo ERROR: OpenCV installation failed!
    echo This might be due to CUDA not being found or compilation errors.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo INSTALLATION COMPLETE!
echo ============================================================
echo.
echo All dependencies installed to: C:\vcpkg\installed\x64-windows
echo.
echo Next step: Build the C++ library
echo   Run: D:\watermarkz\BUILD_SAM2_FINAL.bat
echo.
echo ============================================================
pause
