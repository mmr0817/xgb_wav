@echo off
REM ============================================================
REM  Build xgb_wav image locally (Docker Desktop running + models\w2v_ckpt.pt present)
REM  Result: image xgb_wav:1.0, about 2.3 GB
REM ============================================================
setlocal
cd /d "%~dp0"

docker version >nul 2>nul
if errorlevel 1 goto :nodocker
if not exist "models\w2v_ckpt.pt" goto :noweights

echo Building image xgb_wav:1.0, this can take 5-15 minutes ...
docker build -t xgb_wav:1.0 .
if errorlevel 1 goto :err

echo.
echo Done. Try it:
echo   docker run --rm -v "D:\wavs:/data" -v "%CD%\outputs:/out" xgb_wav:1.0 --input /data --out-dir /out
pause
exit /b 0

:nodocker
echo.
echo [ERROR] Docker not found or not running.
echo         Install/start Docker Desktop first.
pause
exit /b 1

:noweights
echo.
echo [ERROR] models\w2v_ckpt.pt missing - extract the full zip first.
pause
exit /b 1

:err
echo.
echo [ERROR] docker build failed - see messages above.
echo         If it hangs on pulling base image, configure a registry mirror in Docker Desktop.
pause
exit /b 1
