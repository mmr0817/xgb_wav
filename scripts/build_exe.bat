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

REM ---------- 0) Python ----------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found in PATH.
    echo         Install Python 3.12+ ^(64-bit^) and tick "Add to PATH".
    goto :err
)
python -c "import sys; print('[0/5] Python', sys.version.split()[0], '64bit' if sys.maxsize>2**32 else '32bit'); sys.exit(0 if (sys.version_info>=(3,12) and sys.maxsize>2**32) else 1)"
if errorlevel 1 (
    echo [ERROR] 64-bit Python 3.12+ is required.
    echo         xgboost==3.4.1 needs Python ^>= 3.12; 3.11 would install 3.2.0 and fail to load the model.
    goto :err
)

REM ---------- 1) dependencies ----------
if "%SKIP_PIP%"=="1" goto :skip_pip
echo [1/5] Installing dependencies ...
python -m pip install -U pip
if errorlevel 1 goto :err
if "%CUDA%"=="1" (
    python -m pip install torch torchaudio
) else (
    echo       CPU-only torch ^(smaller download, no GPU needed^)
    python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 goto :err
python -m pip install -r requirements.txt pyinstaller
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
python -m xgb_wav.cli --models models --input models\ref_sweep.wav --output build_selftest.csv
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
python -m PyInstaller --noconfirm --clean scripts\xgb_wav.spec
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

:err
echo.
echo [FAILED] see the messages above.
exit /b 1
