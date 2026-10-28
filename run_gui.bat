@echo off
REM ============================================================
REM  双击启动图形界面（出错时窗口会停留，方便看报错）
REM ============================================================
setlocal
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY where py >nul 2>nul
if not defined PY if not errorlevel 1 set "PY=py -3"
if not defined PY where python >nul 2>nul
if not defined PY if not errorlevel 1 set "PY=python"
if not defined PY goto :nopy

echo Starting GUI with: %PY%
%PY% run_gui.py
if errorlevel 1 goto :guierr
exit /b 0

:nopy
echo.
echo [ERROR] Python not found.
echo         Run windows_setup.bat first, or install Python 3.12+ 64-bit
echo         from https://www.python.org/downloads/windows/ and tick Add to PATH.
pause
exit /b 1

:guierr
echo.
echo [ERROR] GUI exited with an error - see messages above.
echo         Tips:
echo           1. run windows_setup.bat to install dependencies
echo           2. run doctor.py and send doctor_report.txt
pause
exit /b 1
