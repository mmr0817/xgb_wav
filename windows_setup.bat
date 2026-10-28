@echo off
REM ============================================================
REM  xgb_wav  Windows 一键安装（双击运行）
REM    1) 找到 Python(64 位, >=3.12)
REM    2) 创建 .venv 虚拟环境
REM    3) 安装依赖（失败自动换清华镜像重试）
REM    4) 运行 doctor.py 自检
REM  纯 ASCII，避免 cmd 代码页问题
REM ============================================================
setlocal
cd /d "%~dp0"

echo ============================================================
echo  xgb_wav Windows setup
echo  dir: %CD%
echo ============================================================

set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY goto :nopython

%PY% -c "import sys; print('[0] Python', sys.version.split()[0], '64bit' if sys.maxsize>2**32 else '32bit'); sys.exit(0 if (sys.version_info>=(3,12) and sys.maxsize>2**32) else 1)"
if errorlevel 1 goto :badpython

echo [1/4] create venv .venv ...
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto :err
) else (
    echo       .venv already exists
)
set "VPY=%CD%\.venv\Scripts\python.exe"

echo [2/4] upgrade pip ...
"%VPY%" -m pip install -U pip
if errorlevel 1 goto :err

echo [3/4] install dependencies, this can take 5-15 min ...
"%VPY%" -m pip install -r requirements.txt
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
echo [ERROR] Python not found in PATH.
echo   Install Python 3.12 or 3.13 (64-bit):
echo     https://www.python.org/downloads/windows/
echo   During install TICK:  Add python.exe to PATH
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
