@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo No se encontro .venv\Scripts\pythonw.exe.
    echo Crea o restaura el entorno virtual del proyecto antes de ejecutar.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" "src\detection.py"
if errorlevel 1 (
    echo.
    echo No se pudo iniciar el programa.
    pause
    exit /b 1
)

endlocal
