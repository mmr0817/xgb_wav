@echo off
REM ============================================================
REM  Double-click to start the GUI.
REM  Window stays open on errors so you can read the message.
REM
REM  IMPORTANT: .venv CANNOT be copied between computers.
REM  (compiled .pyd files are tied to one Python ABI/build)
REM  If .venv is missing or broken, run windows_setup.bat on THIS machine.
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
echo [ERROR] No usable Python environment found on THIS machine.
echo.
echo   Run  windows_setup.bat  once in this folder to create .venv
echo   and install the dependencies (needs Python 3.12 x64).
echo.
echo   Or just use the packaged exe instead:  dist\xgb_wav\xgb_wav.exe
echo   (that folder needs no Python at all)
pause
exit /b 1

:bad_venv
echo.
echo [ERROR] .venv exists but is unusable on this machine.
echo.
echo   Typical cause: the folder was copied from another PC.
echo   A virtualenv cannot be moved: its compiled modules match one
echo   exact Python build (e.g. cp312 .pyd loaded by Python 3.14).
echo.
echo   Fix (pick one):
echo     1) run  windows_setup.bat   - it rebuilds .venv here
echo     2) delete the .venv folder, then run windows_setup.bat
echo     3) ship the exe instead: copy dist\xgb_wav\ (no Python needed)
echo.
pause
exit /b 1

:have_py
echo Starting GUI with: %PY%
%PY% run_gui.py
if errorlevel 1 goto :guierr
exit /b 0

:guierr
echo.
echo [ERROR] GUI exited with an error - see messages above.
echo         Tips:
echo           1. run windows_setup.bat to (re)create .venv
echo           2. run doctor.py and send doctor_report.txt
pause
exit /b 1
