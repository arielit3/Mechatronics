import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts" / "python.exe"
VENV_PYTHONW = VENV / "Scripts" / "pythonw.exe"
APP = ROOT / "src" / "detection.py"
REQUIREMENTS = ROOT / "requirements.txt"
LOGO = ROOT / "src" / "img" / "logo.png"
BASE_PYTHON = Path(sys.executable).with_name("python.exe")

if not BASE_PYTHON.is_file():
    BASE_PYTHON = Path(sys.executable)

events = queue.Queue()
IMPORT_CHECK = "import cv2, ultralytics; from PIL import Image"


def dependencies_ready():
    if not VENV_PYTHON.is_file() or not VENV_PYTHONW.is_file():
        return False

    result = subprocess.run(
        [str(VENV_PYTHON), "-c", IMPORT_CHECK],
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    return result.returncode == 0


def run_command(command, failure_message):
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if result.returncode:
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(
            f"{failure_message}\n\n{details[-3000:]}"
            if details
            else failure_message
        )


def prepare_and_launch():
    if not VENV_PYTHON.is_file():
        events.put(("status", "Creando el entorno virtual..."))
        run_command(
            [str(BASE_PYTHON), "-m", "venv", str(VENV)],
            "No se pudo crear el entorno virtual.",
        )

    if not dependencies_ready():
        events.put(("status", "Descargando e instalando las librerias..."))
        run_command(
            [
                str(VENV_PYTHON),
                "-m",
                "pip",
                "install",
                "--progress-bar",
                "off",
                "-r",
                str(REQUIREMENTS),
            ],
            "No se pudieron instalar las dependencias.",
        )

        if not dependencies_ready():
            raise RuntimeError(
                "La instalacion termino, pero no se pudieron importar "
                "Ultralytics, OpenCV y Pillow."
            )

    events.put(("status", "Iniciando Ixtli..."))
    subprocess.Popen(
        [str(VENV_PYTHONW), str(APP)],
        cwd=ROOT,
    )
    events.put(("success", None))


root = tk.Tk()
root.title("Instalacion de Ixtli")
root.resizable(False, False)
root.geometry("440x155")
root.protocol("WM_DELETE_WINDOW", lambda: None)
if LOGO.is_file():
    logo_image = Image.open(LOGO).convert("RGBA")
    logo_image.thumbnail((64, 64), Image.Resampling.LANCZOS)
    logo_icon = ImageTk.PhotoImage(logo_image)
    root.iconphoto(True, logo_icon)

container = ttk.Frame(root, padding=(24, 20))
container.pack(fill="both", expand=True)

ttk.Label(
    container,
    text="Preparando Ixtli",
    font=("Segoe UI", 15, "bold"),
).pack(anchor="w")

status = tk.StringVar(value="Iniciando instalacion...")
ttk.Label(container, textvariable=status).pack(anchor="w", pady=(10, 8))

progress = ttk.Progressbar(container, mode="indeterminate", length=390)
progress.pack(fill="x")
progress.start(12)


def check_events():
    try:
        while True:
            event, value = events.get_nowait()
            if event == "status":
                status.set(value)
            elif event == "success":
                progress.stop()
                root.destroy()
                return
            elif event == "error":
                progress.stop()
                root.protocol("WM_DELETE_WINDOW", root.destroy)
                messagebox.showerror("Error de instalacion", value, parent=root)
                root.destroy()
                return
    except queue.Empty:
        pass

    root.after(100, check_events)


def run_installer():
    try:
        prepare_and_launch()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        events.put(("error", str(error)))


threading.Thread(target=run_installer, daemon=True).start()
root.after(100, check_events)
root.mainloop()
