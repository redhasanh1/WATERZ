# ======================================================================
# RUN LOCAL TEST IN WSL2 - True 4x Parallel Speedup!
# ======================================================================
#
# This script runs test_local.py logic in WSL2 where:
# - ProcessPoolExecutor WORKS (fork mode, not spawn!)
# - True 4x parallel speedup possible
# - No Windows multiprocessing crashes
#
# Usage:
#   .\RUN_LOCAL_TEST_WSL2.ps1
#
# ======================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LOCAL TEST - WSL2 4-Stream Parallel" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get video path from Windows GUI
Add-Type -AssemblyName System.Windows.Forms
$OpenFileDialog = New-Object System.Windows.Forms.OpenFileDialog
$OpenFileDialog.Title = "Select Video File"
$OpenFileDialog.Filter = "Video Files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All Files (*.*)|*.*"
$OpenFileDialog.InitialDirectory = "D:\watermarkz"

if ($OpenFileDialog.ShowDialog() -eq 'OK') {
    $VideoPath = $OpenFileDialog.FileName
} else {
    Write-Host "❌ No file selected!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Selected: $VideoPath" -ForegroundColor Green
Write-Host ""

# Convert Windows path to WSL2 path
$drive = $VideoPath.Substring(0,1).ToLower()
$pathWithoutDrive = $VideoPath.Substring(2) -replace '\\', '/'
$wslVideoPath = "/mnt/$drive$pathWithoutDrive"

Write-Host "[WSL2] Converting path for Linux..." -ForegroundColor Yellow
Write-Host "  Windows: $VideoPath" -ForegroundColor Gray
Write-Host "  WSL2:    $wslVideoPath" -ForegroundColor Gray
Write-Host ""

Write-Host "[WSL2] Setting up dependencies..." -ForegroundColor Yellow
wsl bash /mnt/d/watermarkz/setup_wsl2_quick.sh
Write-Host ""

Write-Host "[WSL2] Running parallel pipeline with 4 workers..." -ForegroundColor Yellow
Write-Host "  Expected: ~8s first run (warmup), ~1.8s subsequent runs" -ForegroundColor Gray
Write-Host ""

# Run pipeline - single line to avoid line ending issues
wsl bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && PARALLEL_MODE=multiprocessing MAX_PARALLEL_STREAMS=4 python run_local_wsl2.py '$wslVideoPath'"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Result saved in results/ folder" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "❌ Error during processing" -ForegroundColor Red
}
