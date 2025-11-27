$file = "D:\watermarkz\celery_3070_worker.py"
$content = Get-Content $file -Raw

# Fix 1: Initialize all_frames_dir before the if block (add after already_cropped_on_disk = False)
$content = $content -replace "already_cropped_on_disk = False  # Flag to skip redundant memory cropping", "already_cropped_on_disk = False  # Flag to skip redundant memory cropping`n        all_frames_dir = None  # Will be set if using BATCH architecture"

# Fix 2: Replace the broken dir() check with simple variable check
$content = $content -replace "if already_cropped_on_disk and 'all_frames_dir' in dir\(\):", "if already_cropped_on_disk and all_frames_dir:"

Set-Content $file $content -NoNewline
Write-Host "Done - Fixed scope issue"
