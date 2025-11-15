# ======================================================================
# SAM2 WSL2 with Windows GUI Point Selection
# ======================================================================
#
# This script:
# 1. Runs GUI point selection on Windows (native GUI)
# 2. Passes coordinates to WSL2 for fast torch.compile() tracking
#
# Usage:
#   .\RUN_SAM2_WSL2_GUI.ps1 "D:\video.mp4"
#
# ======================================================================

param(
    [string]$VideoPath = ""
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SAM2 WSL2 + Windows GUI Selection" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get video path
if ($VideoPath -eq "") {
    Add-Type -AssemblyName System.Windows.Forms
    $OpenFileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $OpenFileDialog.Title = "Select Video File"
    $OpenFileDialog.Filter = "Video Files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All Files (*.*)|*.*"
    $OpenFileDialog.InitialDirectory = "D:\watermarkz"

    if ($OpenFileDialog.ShowDialog() -eq 'OK') {
        $VideoPath = $OpenFileDialog.FileName
    } else {
        Write-Host "No file selected!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[1/3] Video: $VideoPath" -ForegroundColor Yellow
Write-Host ""

# Run Windows GUI point selector
Write-Host "[2/3] Opening Windows GUI for point selection..." -ForegroundColor Yellow
$pythonPath = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$pointResult = & $pythonPath -c @"
import cv2
import sys

video_path = r'$VideoPath'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print('ERROR: Could not read video')
    sys.exit(1)

clicked_point = [None]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point[0] = (x, y)
        frame_copy = param['frame'].copy()
        cv2.circle(frame_copy, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(frame_copy, (x, y), 10, (0, 255, 0), 2)
        cv2.putText(frame_copy, f'Point: ({x}, {y})', (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Click on object to track (press SPACE when done)', frame_copy)

cv2.namedWindow('Click on object to track (press SPACE when done)', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Click on object to track (press SPACE when done)', mouse_callback, {'frame': frame})
cv2.imshow('Click on object to track (press SPACE when done)', frame)

print('Click on the object you want to track, then press SPACE', file=sys.stderr)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and clicked_point[0] is not None:
        break
    elif key == 27:  # ESC
        cv2.destroyAllWindows()
        sys.exit(1)

cv2.destroyAllWindows()
print(f'{clicked_point[0][0]},{clicked_point[0][1]}')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Could not get point selection" -ForegroundColor Red
    exit 1
}

$point = $pointResult.Trim()
Write-Host "✅ Selected point: $point" -ForegroundColor Green
Write-Host ""

# Convert paths for WSL2
$drive = $VideoPath.Substring(0,1).ToLower()
$pathWithoutDrive = $VideoPath.Substring(2) -replace '\\', '/'
$wslVideoPath = "/mnt/$drive$pathWithoutDrive"

Write-Host "[3/3] Running SAM2 in WSL2 with torch.compile()..." -ForegroundColor Yellow
Write-Host "  Video: $wslVideoPath" -ForegroundColor Gray
Write-Host "  Point: $point" -ForegroundColor Gray
Write-Host ""

# Run in WSL2 with point argument
wsl bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && python test_sam2_wsl2.py '$wslVideoPath' $point"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Complete! Masks saved to D:\watermarkz\temp_sam2_masks\" -ForegroundColor Green
    Write-Host ""
    Write-Host "Opening first mask..." -ForegroundColor Yellow
    Start-Process "D:\watermarkz\temp_sam2_masks\00000.png"
} else {
    Write-Host ""
    Write-Host "❌ Error running SAM2" -ForegroundColor Red
}
