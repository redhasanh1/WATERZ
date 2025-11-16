# ======================================================================
# Process Video with Existing SAM2 Masks (WSL2)
# ======================================================================
#
# Uses existing masks from temp_sam2_masks folder
# Runs ProPainter in WSL2 with all optimizations (NeuFlow, TensorRT, etc.)
#
# Usage:
#   .\RUN_SAM2_WSL2_WITH_MASKS.ps1                    # File picker
#   .\RUN_SAM2_WSL2_WITH_MASKS.ps1 "D:\video.mp4"     # Direct path
#
# ======================================================================

param(
    [string]$VideoPath = ""
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Process Video with Existing SAM2 Masks" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get video path
if ($VideoPath -eq "") {
    Add-Type -AssemblyName System.Windows.Forms
    $OpenFileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $OpenFileDialog.Title = "Select Video File"
    $OpenFileDialog.Filter = "Video Files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All Files (*.*)|*.*"
    $OpenFileDialog.InitialDirectory = "D:\watermarkz\videostotrain"

    if ($OpenFileDialog.ShowDialog() -eq 'OK') {
        $VideoPath = $OpenFileDialog.FileName
    } else {
        Write-Host "No file selected!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[1/3] Video: $VideoPath" -ForegroundColor Yellow
Write-Host ""

# Check masks folder
$MasksFolder = "D:\watermarkz\temp_sam2_masks"
$maskFiles = Get-ChildItem -Path $MasksFolder -Filter *.png -ErrorAction SilentlyContinue

if ($null -eq $maskFiles -or $maskFiles.Count -eq 0) {
    Write-Host "[ERROR] No masks found in temp_sam2_masks!" -ForegroundColor Red
    Write-Host "[ERROR] Run RUN_SAM2_WSL2_GUI.ps1 first to generate masks" -ForegroundColor Red
    exit 1
}

Write-Host "[2/3] Masks: $($maskFiles.Count) files in temp_sam2_masks" -ForegroundColor Yellow
Write-Host ""

# Convert paths for WSL2
$drive = $VideoPath.Substring(0,1).ToLower()
$pathWithoutDrive = $VideoPath.Substring(2) -replace '\\', '/'
$wslVideoPath = "/mnt/$drive$pathWithoutDrive"

$masksDrive = $MasksFolder.Substring(0,1).ToLower()
$masksPathWithoutDrive = $MasksFolder.Substring(2) -replace '\\', '/'
$wslMasksPath = "/mnt/$masksDrive$masksPathWithoutDrive"

Write-Host "[3/3] Running ProPainter in WSL2..." -ForegroundColor Yellow
Write-Host "  Video: $wslVideoPath" -ForegroundColor Gray
Write-Host "  Masks: $wslMasksPath" -ForegroundColor Gray
Write-Host ""

# Run in WSL2
wsl bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && python process_with_masks_wsl2.py '$wslVideoPath' '$wslMasksPath'"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Complete! Output saved to D:\watermarkz\results\" -ForegroundColor Green
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Error running ProPainter" -ForegroundColor Red
}
