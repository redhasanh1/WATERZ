# ======================================================================
# SAM2 WSL2 + Windows GUI + 10fps GPU Optimization
# ======================================================================
#
# This script:
# 1. Preprocesses video to 10fps using NVDEC/NVENC (GPU accelerated)
# 2. Runs GUI point selection on Windows (native GUI)
# 3. Passes coordinates to WSL2 for fast torch.compile() tracking
#
# Optimizations:
# - NVDEC/NVENC hardware encoding (5-6x fewer frames)
# - torch.compile() in WSL2 (111 FPS)
# - BFloat16 precision
#
# Usage:
#   .\RUN_SAM2_WSL2_GUI.ps1                              # Pick video, start from beginning
#   .\RUN_SAM2_WSL2_GUI.ps1 -StartTime "00:02:00"        # Skip 2 minute intro
#   .\RUN_SAM2_WSL2_GUI.ps1 "D:\video.mp4" -StartTime "00:01:30"  # Specific video, skip 1:30
#
# ======================================================================

param(
    [string]$VideoPath = "",
    [string]$StartTime = "00:00:00"  # Skip to this timestamp (HH:MM:SS)
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  SAM2 WSL2 + GUI + 10fps GPU Mode" -ForegroundColor Cyan
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

Write-Host "[1/4] Video: $VideoPath" -ForegroundColor Yellow
Write-Host ""

# Get video info
Write-Host "[2/4] Preprocessing to 10fps with NVDEC/NVENC..." -ForegroundColor Yellow
if ($StartTime -ne "00:00:00") {
    Write-Host "  Starting from: $StartTime (skipping intro)" -ForegroundColor Cyan
}
$pythonPath = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"

# Get FPS and frame count
$videoInfo = & $pythonPath -c @"
import cv2
cap = cv2.VideoCapture(r'$VideoPath')
fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
print(f'{fps},{frames}')
"@

$fps, $totalFrames = $videoInfo.Split(',')
$fps10Frames = [math]::Floor([int]$totalFrames * (10 / [float]$fps))

Write-Host "  Original: $totalFrames frames @ $fps fps" -ForegroundColor Gray
Write-Host "  After 10fps: $fps10Frames frames" -ForegroundColor Gray
Write-Host "  Reduction: $([math]::Round([int]$totalFrames / $fps10Frames, 1))x fewer frames" -ForegroundColor Gray
Write-Host ""

# GPU preprocessing with NVDEC/NVENC (using NVME for speed)
$tempVideo = "C:\water\temp_wsl2_10fps.mp4"

# Ensure NVME temp directory exists
if (-not (Test-Path "C:\water")) {
    New-Item -ItemType Directory -Path "C:\water" -Force | Out-Null
}

# Get FFmpeg from static-ffmpeg
$ffmpegCmd = & $pythonPath -c @"
import os
try:
    from static_ffmpeg import run
    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    print(ffmpeg_path)
except:
    print('ffmpeg')
"@

Write-Host "  Running NVDEC/NVENC preprocessing..." -ForegroundColor Gray

# Run FFmpeg with GPU acceleration (with optional start time)
& $ffmpegCmd -y -hwaccel cuda -ss "$StartTime" -i "$VideoPath" -vf "fps=10" -c:v h264_nvenc -preset p4 -tune hq -b:v 8M -maxrate 10M -bufsize 16M -pix_fmt yuv420p -an "$tempVideo" 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "  Error: GPU preprocessing failed" -ForegroundColor Red
    exit 1
}

$fileSize = [math]::Round((Get-Item $tempVideo).Length / 1MB, 2)
Write-Host "  ✅ Preprocessed: temp_wsl2_10fps.mp4 ($fileSize MB)" -ForegroundColor Green
Write-Host ""

# Run Windows GUI point selector on 10fps video
Write-Host "[3/4] Opening Windows GUI for point selection..." -ForegroundColor Yellow
$pointResult = & $pythonPath -c @"
import cv2
import sys

video_path = r'$tempVideo'
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

# Convert temp video path for WSL2
$drive = $tempVideo.Substring(0,1).ToLower()
$pathWithoutDrive = $tempVideo.Substring(2) -replace '\\', '/'
$wslVideoPath = "/mnt/$drive$pathWithoutDrive"

Write-Host "[4/4] Running SAM2 in WSL2 with torch.compile()..." -ForegroundColor Yellow
Write-Host "  Video (10fps): $wslVideoPath" -ForegroundColor Gray
Write-Host "  Point: $point" -ForegroundColor Gray
Write-Host "  Expected: ~2-3 seconds for $fps10Frames frames @ 111 FPS" -ForegroundColor Gray
Write-Host ""

# Run in WSL2 with point argument
wsl bash -c "cd /mnt/d/watermarkz && source venv_wsl2/bin/activate && python test_sam2_wsl2.py '$wslVideoPath' $point"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Complete! Masks saved to D:\watermarkz\temp_sam2_masks\" -ForegroundColor Green

    # Cleanup temp video
    if (Test-Path $tempVideo) {
        Remove-Item $tempVideo -Force
        Write-Host "✅ Cleaned up temp 10fps video" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "Opening first mask..." -ForegroundColor Yellow
    Start-Process "D:\watermarkz\temp_sam2_masks\00000.png"
} else {
    Write-Host ""
    Write-Host "❌ Error running SAM2" -ForegroundColor Red

    # Cleanup temp video on error too
    if (Test-Path $tempVideo) {
        Remove-Item $tempVideo -Force
    }
}
