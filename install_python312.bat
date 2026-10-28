@echo off
REM ============================================================
REM  Install Python 3.12 (X64) automatically.
REM  Needed because the model was saved by xgboost 3.4.1 (requires Python >= 3.12).
REM
REM  IMPORTANT: always install the x64 build.
REM    - On ARM64 Windows (Parallels on Apple Silicon) winget would pick
REM      the arm64 build by default, but PyTorch has NO win_arm64 wheels,
REM      so pip install torch would fail later.
REM    - x64 Python runs fine on ARM64 Windows (emulated) and all wheels exist.
REM
REM  Your existing Python 3.10 stays untouched.
REM ============================================================
setlocal
set "VER=3.12.10"
set "URL=https://www.python.org/ftp/python/%VER%/python-%VER%-amd64.exe"
set "EXE=%TEMP%\python-%VER%-amd64.exe"

echo ============================================================
echo  Python 3.12 (x64) installer helper
echo  host arch: %PROCESSOR_ARCHITECTURE%
echo ============================================================

py -3.12 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
if not errorlevel 1 goto :already

if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" goto :download
if /i "%PROCESSOR_ARCHITEW6432%"=="ARM64" goto :download

where winget >nul 2>nul
if errorlevel 1 goto :download
echo [1/2] installing via winget, architecture x64 ...
winget install -e --id Python.Python.3.12 --scope user --architecture x64 --silent --accept-package-agreements --accept-source-agreements
py -3.12 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
if not errorlevel 1 goto :ok

:download
echo [2/2] downloading the official x64 installer from python.org ...
echo       %URL%
if exist "%EXE%" del /q "%EXE%"
where curl >nul 2>nul
if errorlevel 1 goto :ps
curl -L --retry 3 --connect-timeout 20 -o "%EXE%" "%URL%"
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
py -3.12 -c "import platform;print('arch:',platform.machine(),'| 64bit:',__import__('sys').maxsize>2**32)"
echo.
echo ============================================================
echo  Python 3.12 installed.
echo  Open a NEW cmd window (so PATH refreshes), then run:
echo     cd /d %~dp0
echo     scripts\build_exe.bat
echo ============================================================
pause
exit /b 0

:already
echo Python 3.12 is already available:
py -3.12 -V
py -3.12 -c "import platform;print('arch:',platform.machine())"
echo.
echo Just run: scripts\build_exe.bat
pause
exit /b 0

:fail
echo.
echo [ERROR] automatic install failed.
echo.
echo   Please install manually (pick the x64 build, NOT arm64):
echo     https://www.python.org/downloads/windows/
echo     -^> "Windows installer (64-bit)"
echo     -^> tick "Add python.exe to PATH"
echo.
echo   Then open a NEW cmd and re-run scripts\build_exe.bat
pause
exit /b 1
