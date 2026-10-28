@echo off
REM ============================================================
REM  xgb_wav  Windows one-click setup (double-click to run)
REM    1) find 64-bit Python >= 3.12
REM    2) create .venv
REM    3) install requirements (retry with Tsinghua mirror on failure)
REM    4) run doctor.py self-check
REM  ASCII only, avoids cmd codepage issues
REM ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo  xgb_wav Windows setup
echo  dir: %CD%
echo ============================================================

set "PY="
py -3.13 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py -3.13"
if not defined PY (
    py -3.12 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=py -3.12"
)
if not defined PY (
    python -c "import sys;sys.exit(0 if (sys.version_info>=(3,12) and sys.maxsize>2**32) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :nopython

%PY% -c "import sys; print('[0/4] Python', sys.version.split()[0], '64bit' if sys.maxsize>2**32 else '32bit', '|', sys.executable)"

echo [1/4] create venv .venv ...
if exist ".venv\Scripts\python.exe" (
    "%CD%\.venv\Scripts\python.exe" -c "import sys;sys.exit(0 if sys.version_info>=(3,12) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo       existing .venv is too old, recreating ...
        rmdir /s /q .venv
    )
)
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto :err
) else (
    echo       .venv already exists ^(Python 3.12+^)
)
set "VPY=%CD%\.venv\Scripts\python.exe"

echo [2/4] upgrade pip ...
REM Optional mirror: set PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
set "PIPOPT="
if not "%PIP_MIRROR%"=="" (
    echo       pip mirror: %PIP_MIRROR%
    set "PIPOPT=-i %PIP_MIRROR%"
)
"%VPY%" -m pip install -U pip %PIPOPT%
if errorlevel 1 goto :err

echo [3/4] install dependencies, this can take 5-15 min ...
"%VPY%" -m pip install -r requirements.txt %PIPOPT%
if errorlevel 1 goto :mirror

echo [4/4] self check ...
"%VPY%" doctor.py
goto :done

:mirror
echo.
echo       default index failed, retry with Tsinghua mirror ...
"%VPY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto :err
echo [4/4] self check ...
"%VPY%" doctor.py
goto :done

:done
echo.
echo ============================================================
echo  Setup finished. Next steps:
echo    GUI  : double-click  run_gui.bat
echo    CLI  : double-click  run_cli.bat  then type a wav folder
echo    EXE  : scripts\build_exe.bat
echo ============================================================
pause
exit /b 0

:nopython
echo.
echo [ERROR] 64-bit Python 3.12+ was not found.
echo.
echo   Detected interpreters:
py -0
echo.
echo   Why 3.12+: the model was saved by xgboost 3.4.1, which requires Python ^>= 3.12.
echo              On 3.10/3.11 pip installs xgboost 3.2.0 and the model cannot be loaded.
echo.
echo   Install 3.12 or newer, then run this script again:
echo     winget install -e --id Python.Python.3.12
echo     or https://www.python.org/downloads/windows/   ^(tick "Add python.exe to PATH"^)
echo.
echo   You do NOT need to uninstall your existing 3.10 - py -3.13 / py -3.12 is used first.
echo.
echo   Or simply double-click  install_python312.bat  in the repo root.
pause
exit /b 1

:badpython
echo.
echo [ERROR] Need 64-bit Python 3.12 or newer.
pause
exit /b 1

:err
echo.
echo [ERROR] see messages above.
pause
exit /b 1
