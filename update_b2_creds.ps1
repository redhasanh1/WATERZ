$file = "D:\watermarkz\celery_3070_worker.py"
$content = Get-Content $file -Raw
$content = $content -replace "B2_KEY_ID = '00539db5c1104b50000000001'", "B2_KEY_ID = '00539db5c1104b50000000002'"
$content = $content -replace "B2_APPLICATION_KEY = 'K005VEORbg6RcsRad3jZPr9n4Fp7jWU'", "B2_APPLICATION_KEY = 'K005HJKUP7ahSNJ1wgQHDDJ+uEATiU4'"
Set-Content $file $content -NoNewline
Write-Host "Done"
