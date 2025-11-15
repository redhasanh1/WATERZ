# Build SAM2 TensorRT C++ Library with Visual Studio
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Building SAM2 TensorRT C++ Library" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Change to sam2_trt_inference directory
Set-Location "D:\watermarkz\sam2_trt_inference"

# Check for engines
if (-not (Test-Path "engines\sam2_encoder_fp16.engine")) {
    Write-Host "ERROR: Encoder engine not found!" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "engines\sam2_decoder_fp16.engine")) {
    Write-Host "ERROR: Decoder engine not found!" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] TensorRT engines found" -ForegroundColor Green
Write-Host ""

# Find Visual Studio
$vsPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
)

$vsPath = $null
foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        $vsPath = $path
        Write-Host "[1/4] Found Visual Studio: $path" -ForegroundColor Green
        break
    }
}

if ($null -eq $vsPath) {
    Write-Host ""
    Write-Host "ERROR: Could not find Visual Studio 2022!" -ForegroundColor Red
    Write-Host "Please install Visual Studio 2022 Build Tools or Community Edition" -ForegroundColor Yellow
    Write-Host "Download: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Clean previous build
Write-Host "[2/4] Cleaning previous build..." -ForegroundColor Yellow
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
Write-Host "Done" -ForegroundColor Green
Write-Host ""

# Configure CMake
Write-Host "[3/4] Configuring CMake with vcpkg..." -ForegroundColor Yellow

# Set up VS environment and run CMake
$env:Path = "C:\Program Files\CMake\bin;$env:Path"
& cmd /c "`"$vsPath`" -arch=x64 -host_arch=x64 && cmake -B build -G `"Visual Studio 17 2022`" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake -DCMAKE_BUILD_TYPE=Release"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: CMake configuration failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Build
Write-Host "[4/4] Building Release configuration..." -ForegroundColor Yellow
Write-Host "This may take 10-15 minutes..." -ForegroundColor Gray
Write-Host ""

cmake --build build --config Release

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "BUILD COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path "build\Release\trt_sam2_infer.dll") {
    Write-Host "[SUCCESS] DLL created:" -ForegroundColor Green
    Get-ChildItem "build\Release\trt_sam2_infer.dll"
    Write-Host ""
    Write-Host "Next: Test the Python wrapper" -ForegroundColor Yellow
    Write-Host "  python D:\watermarkz\sam2_trt_predictor_fast.py" -ForegroundColor Gray
} else {
    Write-Host "[WARN] DLL not found in expected location" -ForegroundColor Yellow
    Write-Host "Searching..." -ForegroundColor Gray
    Get-ChildItem -Recurse -Filter "*.dll" "build"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
