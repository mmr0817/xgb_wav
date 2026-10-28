@echo off
REM ============================================================
REM  Run batch detection with Docker (Windows, Docker Desktop required)
REM  Double-click, drag a wav folder into this window, press Enter
REM  Result: outputs\predictions.csv in this folder
REM ============================================================
setlocal
cd /d "%~dp0"

docker version >nul 2>nul
if errorlevel 1 goto :nodocker

set "IMAGE=xgb_wav:1.0"
set "TARGET=%~1"
if "%TARGET%"=="" set /p TARGET=Enter wav folder path, e.g. D:\wavs : 
if "%TARGET%"=="" goto :noinput
if not exist "%TARGET%" goto :notfound
if not exist "outputs" mkdir "outputs"

echo.
echo Image : %IMAGE%
echo Input : %TARGET%
echo Output: %CD%\outputs
echo.

docker image inspect %IMAGE% >nul 2>nul
if errorlevel 1 (
    echo [info] image not found locally, trying to pull ...
    docker pull %IMAGE%
    if errorlevel 1 echo [warn] pull failed - build it first with docker_build.bat
)

docker run --rm -v "%TARGET%:/data" -v "%CD%\outputs:/out" %IMAGE% --input /data --out-dir /out
if errorlevel 1 goto :runerr

echo.
echo Done. CSV: %CD%\outputs\predictions.csv
pause
exit /b 0

:nodocker
echo.
echo [ERROR] Docker not found or not running.
echo         Install Docker Desktop: https://www.docker.com/products/docker-desktop/
echo         Then start it and wait until the whale icon stops animating.
pause
exit /b 1

:noinput
echo.
echo [ERROR] No input folder given.
pause
exit /b 1

:notfound
echo.
echo [ERROR] Folder not found: %TARGET%
pause
exit /b 1

:runerr
echo.
echo [ERROR] docker run failed - see messages above.
pause
exit /b 1
