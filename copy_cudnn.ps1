Set-Location "D:\watermarkz\TensorRT-10.13.3.9\lib"

Write-Host "Copying cuDNN 9 DLLs to cuDNN 8 names..."

Copy-Item "cudnn64_9.dll" "cudnn64_8.dll" -Force
Copy-Item "cudnn_adv64_9.dll" "cudnn_adv64_8.dll" -Force
Copy-Item "cudnn_cnn64_9.dll" "cudnn_cnn64_8.dll" -Force
Copy-Item "cudnn_ops64_9.dll" "cudnn_ops64_8.dll" -Force
Copy-Item "cudnn_engines_precompiled64_9.dll" "cudnn_engines_precompiled64_8.dll" -Force
Copy-Item "cudnn_engines_runtime_compiled64_9.dll" "cudnn_engines_runtime_compiled64_8.dll" -Force
Copy-Item "cudnn_graph64_9.dll" "cudnn_graph64_8.dll" -Force
Copy-Item "cudnn_heuristic64_9.dll" "cudnn_heuristic64_8.dll" -Force

Write-Host ""
Write-Host "cuDNN 8 compatibility DLLs created:"
Get-ChildItem "cudnn*_8.dll" | Select-Object Name, @{N='Size(MB)';E={[math]::Round($_.Length/1MB,2)}}
