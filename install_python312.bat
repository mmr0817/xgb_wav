@echo off
REM ============================================================
REM  Install Python 3.12 (64-bit) automatically.
REM  Needed because the model was saved by xgboost 3.4.1 (requires Python >= 3.12).
REM
REM  Attempts, in order:
REM    1) winget (per-user, silent)
REM    2) download python-3.12.10-amd64.exe from python.org and run it
REM
REM  Your existing Python 3.10 stays untouched.
REM ============================================================
setlocal
set "VER=3.12.10"
set "URL=https://www.python.org/ftp/python/%VER%/python-%VER%-amd64.exe"
set "EXE=%TEMP%\python-%VER%-amd64.exe"

echo ============================================================
echo  Python 3.12 installer helper
echo ============================================================

py -3.12 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
if not errorlevel 1 goto :already

where winget >nul 2>nul
if errorlevel 1 goto :download
echo [1/2] installing via winget ...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
py -3.12 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
if not errorlevel 1 goto :ok

:download
echo [2/2] winget unavailable or failed - downloading from python.org ...
echo       %URL%
if exist "%EXE%" del /q "%EXE%"
where curl >nul 2>nul
if errorlevel 1 goto :ps
curl -L --retry 3 -o "%EXE%" "%URL%"
goto :run
:ps
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%EXE%'"
:run
if not exist "%EXE%" goto :fail
echo       installing silently: per-user, add to PATH, include py launcher ...
"%EXE%" /passive InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0 Include_pip=1
if errorlevel 1 goto :fail

:ok
echo.
py -3.12 -V
echo.
echo ============================================================
echo  Python 3.12 installed.
echo  Open a NEW cmd window, then run:
echo     cd /d %~dp0
echo     scripts\build_exe.bat
echo ============================================================
pause
exit /b 0

:already
echo Python 3.12 is already available:
py -3.12 -V
echo.
echo Just run: scripts\build_exe.bat
pause
exit /b 0

:fail
echo.
echo [ERROR] automatic install failed.
echo   Please install manually:
echo     https://www.python.org/downloads/windows/
echo   Pick "Windows installer (64-bit)" and tick "Add python.exe to PATH".
pause
exit /b 1
