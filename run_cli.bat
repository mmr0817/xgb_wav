@echo off
REM ============================================================
REM  Batch detection: double-click, drag a wav folder in, press Enter
REM  CSV goes to outputs\cli\
REM ============================================================
setlocal
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY py -3.13 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
if not defined PY if not errorlevel 1 set "PY=py -3.13"
if not defined PY py -3.12 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
if not defined PY if not errorlevel 1 set "PY=py -3.12"
if not defined PY python -c "import sys;sys.exit(0 if (sys.version_info>=(3,12) and sys.maxsize>2**32) else 1)" >nul 2>nul
if not defined PY if not errorlevel 1 set "PY=python"
if not defined PY goto :nopy

set "TARGET=%~1"
if "%TARGET%"=="" set /p TARGET=Enter wav folder path, e.g. D:\wavs : 
if "%TARGET%"=="" goto :noinput

echo.
echo Python : %PY%
echo Input  : %TARGET%
echo.
%PY% test.py --input "%TARGET%" --out-dir "%~dp0outputs\cli"
if errorlevel 1 goto :clierr
echo.
echo Done. CSV is in outputs\cli\
pause
exit /b 0

:nopy
echo.
echo [ERROR] Python not found. Run windows_setup.bat first.
pause
exit /b 1

:noinput
echo.
echo [ERROR] No input path given.
pause
exit /b 1

:clierr
echo.
echo [ERROR] Detection failed - see messages above.
echo         Tips: run doctor.py and send doctor_report.txt
pause
exit /b 1
