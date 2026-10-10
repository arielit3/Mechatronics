@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" if exist ".venv\Scripts\pythonw.exe" (
    ".venv\Scripts\python.exe" -c "import cv2, ultralytics; from PIL import Image" >nul 2>nul
    if not errorlevel 1 (
        start "" ".venv\Scripts\pythonw.exe" "src\detection.py"
        exit /b
    )
)

set "PYTHON="
where py >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON=%%I"
)

if not defined PYTHON (
    where python >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON=%%I"
    )
)

if not defined PYTHON (
    echo No se encontro Python. Instala Python 3 y vuelve a intentarlo.
    pause
    exit /b 1
)

for %%I in ("%PYTHON%") do set "PYTHONW=%%~dpIpythonw.exe"
if not exist "%PYTHONW%" (
    echo No se encontro pythonw.exe junto a Python.
    echo Reinstala Python incluyendo Tcl/Tk y el lanzador de ventanas.
    pause
    exit /b 1
)

"%PYTHON%" -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo La instalacion de Python no incluye Tkinter, necesario para mostrar el progreso.
    pause
    exit /b 1
)

start "" "%PYTHONW%" "installer.py"
endlocal
exit /b
