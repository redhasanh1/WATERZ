@echo off
echo ========================================
echo Installing Compatible PyTorch + Torchvision
echo ========================================
echo.
echo The issue: PyTorch 2.4.1 is too old for torchvision 0.19.1
echo Solution: Use PyTorch 2.5.1 + torchvision 0.20.1 (compatible pair)
echo.

echo Uninstalling current versions...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch 2.5.1 + torchvision 0.20.1 with CUDA 12.4...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo.
echo ========================================
echo Verifying installation...
echo ========================================
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

echo.
echo Testing watermark import...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -c "import sys; sys.path.insert(0, 'D:/watermarkz/faster-propainter-main'); from watermark import pipeline; print('SUCCESS: watermark module imports correctly!')"

echo.
pause
