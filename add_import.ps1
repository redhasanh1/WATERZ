$file = "D:\watermarkz\celery_3070_worker.py"
$content = Get-Content $file -Raw
$content = $content -replace "import zipfile`r?`nimport requests", "import zipfile`nimport subprocess`nimport requests"
Set-Content $file $content -NoNewline
Write-Host "Done"
