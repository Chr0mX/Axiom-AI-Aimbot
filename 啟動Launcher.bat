@echo off
chcp 65001 >nul
cd /d "%~dp0"

net session >nul 2>&1
if %errorlevel% NEQ 0 (
	powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'cmd.exe' -Verb RunAs -ArgumentList '/c','\""%~f0"\" %*'" 
	exit /b
)

REM === Check and install Visual C++ Redistributable ===
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version >nul 2>&1
if %errorlevel% EQU 0 (
	echo [Axiom] Visual C++ Redistributable is already installed, skipping.
	goto :VC_DONE
)

if not exist "%~dp0src\VC_redist.x64.exe" (
	echo [Axiom] Warning: Visual C++ Redistributable not detected, and installer not found.
	echo [Axiom] If the program fails to start, please download and install from https://aka.ms/vs/17/release/vc_redist.x64.exe
	goto :VC_DONE
)

echo [Axiom] Installing Visual C++ Redistributable, please wait...
"%~dp0src\VC_redist.x64.exe" /install /quiet /norestart
if %errorlevel% NEQ 0 (
	echo [Axiom] Visual C++ Redistributable installation failed, please install manually.
	echo [Axiom] Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
	pause
	exit /b 1
)
echo [Axiom] Visual C++ Redistributable installation completed.

:VC_DONE
echo [Axiom] Checking Python dependencies...
src\python\python.exe src\tools\bootstrap_dependencies.py
if %errorlevel% NEQ 0 (
	echo [Axiom] Dependency bootstrap failed. Please check your network and Python runtime.
	pause
	exit /b 1
)

src\python\python.exe src\main.py
