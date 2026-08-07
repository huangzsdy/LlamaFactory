@echo off
REM 创建 OpenCompass 离线运行环境 (Windows)
REM 使用方法: 双击运行或 cmd /c create_venv.bat

echo ==========================================
echo 创建 OpenCompass 虚拟环境
echo ==========================================

REM 检查Python版本
python --version
if errorlevel 1 (
    echo 未找到Python，请先安装Python 3.10+
    pause
    exit /b 1
)

REM 创建虚拟环境
echo [1/3] 创建虚拟环境 opencompass_env...
python -m venv opencompass_env

REM 激活虚拟环境
echo [2/3] 安装依赖...
call opencompass_env\Scripts\pip.exe install --upgrade pip

REM 安装依赖
call opencompass_env\Scripts\pip.exe install torch
call opencompass_env\Scripts\pip.exe install transformers
call opencompass_env\Scripts\pip.exe install datasets
call opencompass_env\Scripts\pip.exe install pandas
call opencompass_env\Scripts\pip.exe install openpyxl
call opencompass_env\Scripts\pip.exe install sacrebleu
call opencompass_env\Scripts\pip.exe install rouge
call opencompass_env\Scripts\pip.exe install opencompass

echo ==========================================
echo 虚拟环境创建完成!
echo ==========================================
echo.
echo 激活环境: opencompass_env\Scripts\activate.bat
echo 运行评测: python run.py --config ..\datasets\my_custom_all.py
echo.
pause
