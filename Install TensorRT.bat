@echo off
chcp 65001 >nul
cd /d "%~dp0"
net session >nul 2>&1
if %errorlevel% NEQ 0 (
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'cmd.exe' -Verb RunAs -ArgumentList '/c','\""%~f0"\" %*'"
	exit /b
)

echo [Axiom] Installing TensorRT / onnxruntime-gpu packages...
echo [Axiom] Packages will be written to: %LOCALAPPDATA%\AxiomAI\site-packages
echo.

src\python\python.exe src\install_tensorrt_local.py
if %errorlevel% NEQ 0 (
	echo.
	echo [Axiom] Installation encountered an error. See output above.
	pause
	exit /b 1
)

echo.
echo [Axiom] Done. Restart Axiom and set the inference backend to CUDA or TensorRT.
pause
