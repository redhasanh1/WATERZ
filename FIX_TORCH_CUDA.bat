@echo off
echo ========================================
echo Reinstalling PyTorch with CUDA support
echo ========================================
echo.
echo The torchvision downgrade installed CPU-only PyTorch 2.4.1
echo We need PyTorch 2.4.1+cu124 for CUDA support
echo.

echo Uninstalling CPU-only PyTorch...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip uninstall -y torch torchvision

echo.
echo Installing PyTorch 2.4.1 + torchvision 0.19.1 with CUDA 12.4...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

echo.
echo ========================================
echo Verifying CUDA support...
echo ========================================
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo.
pause
