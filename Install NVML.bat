@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [Axiom] Installing nvidia-ml-py (NVIDIA GPU monitoring for the status panel)...
echo [Axiom] Package will be written to: src\python\Lib\site-packages
echo.

src\python\python.exe -m pip install --upgrade nvidia-ml-py
if %errorlevel% NEQ 0 (
	echo.
	echo [Axiom] Installation encountered an error. See output above.
	pause
	exit /b 1
)

echo.
echo [Axiom] Done. Restart Axiom, then turn on Visuals ^> Status Panel ^> GPU Monitoring.
pause
