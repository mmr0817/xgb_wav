@echo off
REM ============================================================
REM  xgb_wav  Windows build script (PyInstaller)
REM
REM  Usage (from repo root):
REM      scripts\build_exe.bat
REM
REM  Output:
REM      dist\xgb_wav\xgb_wav.exe   (+ models\ copied next to it)
REM
REM  Requires 64-bit Python 3.12 or newer:
REM      the model was saved by xgboost 3.4.1, which needs Python >= 3.12.
REM  The script auto-detects: py -3.13 -> py -3.12 -> python (if it is 3.12+).
REM  Having Python 3.10 installed as well is fine.
REM
REM  Optional flags (set before running):
REM      set SKIP_PIP=1     skip pip install (deps already present)
REM      set NO_SELFTEST=1  skip the pre-build inference self-test
REM      set CUDA=1         install CUDA build of torch (default: CPU)
REM
REM  NOTE: keep this file ASCII-only. cmd.exe reads .bat in the OEM
REM        codepage, non-ASCII text can break parsing on some systems.
REM ============================================================
setlocal
cd /d "%~dp0\.."

echo ============================================================
echo  xgb_wav build   ^|  %CD%
echo ============================================================

REM ---------- 0) find a 64-bit Python >= 3.12 ----------
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

%PY% -c "import sys;print('[0/5] Python', sys.version.split()[0], '64bit' if sys.maxsize>2**32 else '32bit', '|', sys.executable)"

REM ---------- 1) dependencies ----------
if "%SKIP_PIP%"=="1" goto :skip_pip
echo [1/5] Installing dependencies ...

REM Optional mirror, e.g.:
REM    set PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
set "PIPOPT="
if not "%PIP_MIRROR%"=="" (
    echo       pip mirror: %PIP_MIRROR%
    set "PIPOPT=-i %PIP_MIRROR%"
)
%PY% -m pip install -U pip %PIPOPT%
if errorlevel 1 goto :err

if "%CUDA%"=="1" goto :pip_cuda
if not "%PIP_MIRROR%"=="" goto :pip_torch_mirror
echo       CPU-only torch from pytorch CPU index
%PY% -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
goto :pip_torch_done
:pip_torch_mirror
echo       CPU-only torch from mirror ^(PyPI win_amd64 wheel is CPU-only^)
%PY% -m pip install torch torchaudio %PIPOPT%
goto :pip_torch_done
:pip_cuda
echo       CUDA torch from index/PyPI
%PY% -m pip install torch torchaudio %PIPOPT%
:pip_torch_done
if errorlevel 1 goto :err
%PY% -m pip install -r requirements.txt pyinstaller %PIPOPT%
if errorlevel 1 goto :err
:skip_pip

REM ---------- 2) model files ----------
echo [2/5] Checking models\ ...
call :check models\detector.json       || goto :err
call :check models\fusion.json         || goto :err
call :check models\xgb_pipeline.joblib || goto :err
call :check models\w2v_ckpt.pt         || goto :err
call :check models\ref_sweep.wav       || goto :err

REM ---------- 3) inference self-test ----------
if "%NO_SELFTEST%"=="1" goto :skip_test
echo [3/5] Self-test: CLI inference on models\ref_sweep.wav ...
set "PYTHONPATH=%CD%\src"
%PY% -m xgb_wav.cli --models models --input models\ref_sweep.wav --output build_selftest.csv
if errorlevel 1 (
    echo [ERROR] self-test failed - fix the runtime BEFORE packaging.
    goto :err
)
echo       self-test OK ^(result: build_selftest.csv^)
goto :after_test
:skip_test
echo [3/5] Self-test skipped
:after_test

REM ---------- 4) package ----------
echo [4/5] Packaging with PyInstaller ...
%PY% -m PyInstaller --noconfirm --clean scripts\xgb_wav.spec
if errorlevel 1 goto :err

REM ---------- 5) copy models ----------
echo [5/5] Copying models\ next to the exe ...
xcopy /E /I /Y models "dist\xgb_wav\models" >nul
if errorlevel 1 (
    echo   [WARN] copy failed, please copy models\ manually.
) else (
    echo   models\ copied
)

echo.
echo ============================================================
echo  DONE
echo    exe    : dist\xgb_wav\xgb_wav.exe
echo    models : dist\xgb_wav\models\
if exist "dist\xgb_wav\models\w2v_ckpt.pt" (
    echo    offline: YES ^(weights ship next to the exe, no download^)
) else (
    echo    [WARN] w2v_ckpt.pt missing next to the exe - app will fail
)
echo.
echo  Ship the WHOLE folder dist\xgb_wav\ -, the exe alone will not run.
echo  First start takes 5-15 s ^(loading torch^). See docs\BUILD_WINDOWS.md
echo ============================================================
goto :eof

:check
if exist "%~1" (
    echo   ok       %~1
    exit /b 0
)
echo   MISSING  %~1
echo [ERROR] models\ is incomplete. Copy the trained model folder here first.
exit /b 1

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
echo   Or simply double-click  install_python312.bat  in the repo root.
echo.
echo   You do NOT need to uninstall 3.10 - this script prefers py -3.13 / py -3.12.
echo   Alternative: use the Docker image ^(windows_setup.bat is not required^).
echo.
pause
exit /b 1

:err
echo.
echo [FAILED] see the messages above.
exit /b 1
