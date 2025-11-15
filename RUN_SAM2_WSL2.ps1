# ======================================================================
# SAM2 WSL2 - Maximum Performance with torch.compile()
# ======================================================================
#
# This launches SAM2 in WSL2 Ubuntu where torch.compile() works!
#
# Why WSL2?
#   - torch.compile() BROKEN on Windows (Triton compiler issues)
#   - torch.compile() WORKS on Linux (Triton generates optimized kernels)
#   - WSL2 GPU passthrough is within 1% of native Linux performance
#
# Usage:
#   .\RUN_SAM2_WSL2.ps1                    # File picker
#   .\RUN_SAM2_WSL2.ps1 "D:\video.mp4"     # Direct path
#
# ======================================================================

param(
    [string]$VideoPath = ""
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SAM2 WSL2 - torch.compile() Edition" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check WSL2 installation
Write-Host "[1/3] Checking WSL2..." -ForegroundColor Yellow
$wslCheck = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ WSL2 not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Install WSL2 with:" -ForegroundColor Yellow
    Write-Host "  wsl --install -d Ubuntu-22.04" -ForegroundColor White
    Write-Host ""
    Write-Host "Then see SETUP_WSL2.md for full setup instructions" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ WSL2 found" -ForegroundColor Green

# Convert Windows path to WSL2 path
$wslPath = ""
if ($VideoPath -ne "") {
    Write-Host "[2/3] Converting path to WSL2..." -ForegroundColor Yellow
    # Convert D:\watermarkz\video.mp4 -> /mnt/d/watermarkz/video.mp4
    $drive = $VideoPath.Substring(0,1).ToLower()
    $pathWithoutDrive = $VideoPath.Substring(2) -replace '\\', '/'
    $wslPath = "/mnt/$drive$pathWithoutDrive"
    Write-Host "  Windows: $VideoPath" -ForegroundColor Gray
    Write-Host "  WSL2:    $wslPath" -ForegroundColor Gray
} else {
    Write-Host "[2/3] No video specified, will use interactive prompt" -ForegroundColor Yellow
}

# Launch in WSL2
Write-Host "[3/3] Launching SAM2 in WSL2..." -ForegroundColor Yellow
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Starting torch.compile() optimized SAM2..." -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

if ($wslPath -ne "") {
    wsl bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && python test_sam2_wsl2.py '$wslPath'"
} else {
    wsl bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && python test_sam2_wsl2.py"
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ SAM2 WSL2 completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "❌ Error running SAM2 in WSL2" -ForegroundColor Red
    Write-Host "See SETUP_WSL2.md for troubleshooting" -ForegroundColor Yellow
}
