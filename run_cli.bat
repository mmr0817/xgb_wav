@echo off
REM ============================================================
REM  Batch detection: double-click, drag a wav folder in, press Enter.
REM  CSV goes to outputs\cli\
REM
REM  NOTE: .venv cannot be copied between computers; if it is broken
REM        run windows_setup.bat on this machine.
REM ============================================================
setlocal
cd /d "%~dp0"

set "PY="
set "VPY=%CD%\.venv\Scripts\python.exe"

if not exist "%VPY%" goto :try_system
"%VPY%" -c "import numpy, torch, xgboost" >nul 2>nul
if errorlevel 1 goto :bad_venv
set "PY=%VPY%"
goto :have_py

:try_system
py -3.13 -c "import numpy, torch, xgboost" >nul 2>nul
if not errorlevel 1 set "PY=py -3.13"
if defined PY goto :have_py
py -3.12 -c "import numpy, torch, xgboost" >nul 2>nul
if not errorlevel 1 set "PY=py -3.12"
if defined PY goto :have_py
python -c "import sys, numpy, torch, xgboost; sys.exit(0 if sys.version_info>=(3,12) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto :have_py
echo.
echo [ERROR] No usable Python environment. Run windows_setup.bat first.
echo         Or ship dist\xgb_wav\ (the exe needs no Python).
pause
exit /b 1

:bad_venv
echo.
echo [ERROR] .venv cannot be used on this machine (probably copied from another PC).
echo         Run windows_setup.bat to rebuild it here, or delete .venv first.
pause
exit /b 1

:have_py
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
