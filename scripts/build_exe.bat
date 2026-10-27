@echo off
REM ============================================================
REM  xgb_wav  Windows 打包脚本
REM  用法：在仓库根目录双击或在命令行执行 scripts\build_exe.bat
REM  产物：dist\xgb_wav\xgb_wav.exe（models\ 会被复制到 exe 同级）
REM ============================================================
setlocal
cd /d "%~dp0\.."

echo [1/4] 检查 Python
python --version || goto :err

echo [2/4] 安装依赖（含 PyInstaller）
python -m pip install -U pip
python -m pip install -r requirements.txt pyinstaller || goto :err

echo [3/4] 打包
python -m PyInstaller --noconfirm --clean scripts\xgb_wav.spec || goto :err

echo [4/4] 复制模型到 dist
if exist "models\detector.json" (
    xcopy /E /I /Y models "dist\xgb_wav\models" >nul
    echo   models\ 已复制
) else (
    echo   [警告] 未找到 models\detector.json，请先运行 scripts\train_production.py
)

echo.
echo 完成： dist\xgb_wav\xgb_wav.exe
echo 注意：首次运行会下载 wav2vec2 预训练权重（约 360 MB），请保证网络可用。
goto :eof

:err
echo.
echo [失败] 请检查上方的错误信息。
exit /b 1
