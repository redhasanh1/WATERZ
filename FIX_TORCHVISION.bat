@echo off
echo ========================================
echo Fixing torchvision compatibility
echo ========================================
echo.
echo Uninstalling current torchvision...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip uninstall -y torchvision

echo.
echo Installing compatible torchvision 0.19.1...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip install torchvision==0.19.1

echo.
echo ========================================
echo Testing import...
echo ========================================
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -c "import torch; import torchvision; print('SUCCESS: PyTorch', torch.__version__, '+ torchvision', torchvision.__version__)"

echo.
pause
