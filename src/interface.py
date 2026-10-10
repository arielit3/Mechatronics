import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

RUTA_LOGO = Path(__file__).resolve().parent / "img" / "logo.png"

UMBRAL_INICIAL = 0.35
UMBRAL_MINIMO = 0.05
UMBRAL_MAXIMO = 0.95
PASO_UMBRAL = 0.05


def calcular_umbral(umbral_actual, cambio):
    valor = max(UMBRAL_MINIMO, min(UMBRAL_MAXIMO, umbral_actual + cambio))
    return round(valor, 2)


def ejecutar_interfaz(procesar_camara, clases_permitidas):
    raiz = tk.Tk()
    raiz.title("Ixtli - Deteccion en tiempo real")
    raiz.geometry("1024x620")
    raiz.minsize(980, 600)
    raiz.configure(bg="#111827")

    logo = Image.open(RUTA_LOGO).convert("RGBA")
    logo.thumbnail((88, 88), Image.Resampling.LANCZOS)
    imagen_logo = ImageTk.PhotoImage(logo)
    raiz.iconphoto(True, imagen_logo)

    estado_umbral = {"valor": UMBRAL_INICIAL}
    bloqueo_umbral = threading.Lock()
    estado_camara = {"indice": None}
    bloqueo_camara = threading.Lock()
    cola_fotogramas = queue.Queue(maxsize=1)
    cola_errores = queue.Queue()
    cola_camaras = queue.Queue(maxsize=1)
    detener = threading.Event()

    panel = tk.Frame(raiz, bg="#1f2937", width=260, padx=20, pady=24)
    panel.pack(side=tk.LEFT, fill=tk.Y)
    panel.pack_propagate(False)

    tk.Label(
        panel,
        image=imagen_logo,
        bg="#1f2937",
    ).pack(anchor=tk.CENTER, pady=(0, 8))
    tk.Label(
        panel,
        text="IXTLI",
        font=("Segoe UI", 22, "bold"),
        fg="#f9fafb",
        bg="#1f2937",
    ).pack(anchor=tk.W)
    tk.Label(
        panel,
        text="Panel de deteccion",
        font=("Segoe UI", 10),
        fg="#9ca3af",
        bg="#1f2937",
    ).pack(anchor=tk.W, pady=(0, 18))

    tk.Label(
        panel,
        text="Umbral de confianza",
        font=("Segoe UI", 12, "bold"),
        fg="#f9fafb",
        bg="#1f2937",
    ).pack(anchor=tk.W)
    tk.Label(
        panel,
        text="Confianza minima para mostrar una deteccion",
        wraplength=215,
        justify=tk.LEFT,
        font=("Segoe UI", 9),
        fg="#9ca3af",
        bg="#1f2937",
    ).pack(anchor=tk.W, pady=(5, 12))

    etiqueta_umbral = tk.StringVar(value=f"{UMBRAL_INICIAL:.0%}")
    controles_umbral = tk.Frame(panel, bg="#1f2937")
    controles_umbral.pack(fill=tk.X)
    boton_disminuir = tk.Button(
        controles_umbral,
        text="−",
        font=("Segoe UI", 18, "bold"),
        width=3,
        command=lambda: ajustar_umbral(-PASO_UMBRAL),
    )
    boton_disminuir.pack(side=tk.LEFT)
    tk.Label(
        controles_umbral,
        textvariable=etiqueta_umbral,
        font=("Segoe UI", 18, "bold"),
        width=6,
        fg="#f9fafb",
        bg="#1f2937",
    ).pack(side=tk.LEFT, expand=True)
    boton_aumentar = tk.Button(
        controles_umbral,
        text="+",
        font=("Segoe UI", 18, "bold"),
        width=3,
        command=lambda: ajustar_umbral(PASO_UMBRAL),
    )
    boton_aumentar.pack(side=tk.RIGHT)

    tk.Label(
        panel,
        text="Camara",
        font=("Segoe UI", 11, "bold"),
        fg="#f9fafb",
        bg="#1f2937",
    ).pack(anchor=tk.W, pady=(18, 5))
    seleccion_camara = tk.StringVar(value="Buscando camaras...")
    selector_camara = ttk.Combobox(
        panel,
        textvariable=seleccion_camara,
        state="disabled",
        width=24,
    )
    selector_camara.pack(fill=tk.X)

    def cambiar_camara(evento):
        indice = indices_por_nombre.get(evento.widget.get())
        if indice is not None:
            with bloqueo_camara:
                estado_camara["indice"] = indice

    indices_por_nombre = {}
    selector_camara.bind("<<ComboboxSelected>>", cambiar_camara)

    def ajustar_umbral(cambio):
        with bloqueo_umbral:
            estado_umbral["valor"] = calcular_umbral(
                estado_umbral["valor"],
                cambio,
            )
            valor = estado_umbral["valor"]

        etiqueta_umbral.set(f"{valor:.0%}")
        boton_disminuir.configure(
            state=tk.DISABLED if valor <= UMBRAL_MINIMO else tk.NORMAL
        )
        boton_aumentar.configure(
            state=tk.DISABLED if valor >= UMBRAL_MAXIMO else tk.NORMAL
        )

    tk.Label(
        panel,
        text="Objetos vistos (total)",
        font=("Segoe UI", 12, "bold"),
        fg="#f9fafb",
        bg="#1f2937",
    ).pack(anchor=tk.W, pady=(32, 12))

    variables_conteo = {}
    for etiqueta in clases_permitidas.values():
        fila = tk.Frame(panel, bg="#1f2937")
        fila.pack(fill=tk.X, pady=3)
        tk.Label(
            fila,
            text=etiqueta,
            font=("Segoe UI", 10),
            fg="#d1d5db",
            bg="#1f2937",
        ).pack(side=tk.LEFT)
        variable = tk.StringVar(value="0")
        variables_conteo[etiqueta] = variable
        tk.Label(
            fila,
            textvariable=variable,
            font=("Segoe UI", 10, "bold"),
            fg="#f9fafb",
            bg="#1f2937",
        ).pack(side=tk.RIGHT)

    tk.Label(
        panel,
        text="Acumulado de objetos unicos detectados durante esta sesion.",
        wraplength=215,
        justify=tk.LEFT,
        font=("Segoe UI", 9),
        fg="#9ca3af",
        bg="#1f2937",
    ).pack(anchor=tk.W, pady=(8, 0))

    contenido = tk.Frame(raiz, bg="#111827", padx=20, pady=20)
    contenido.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    etiqueta_fps = tk.StringVar(value="FPS: --")
    tk.Label(
        contenido,
        textvariable=etiqueta_fps,
        font=("Segoe UI", 11, "bold"),
        fg="#86efac",
        bg="#111827",
    ).pack(anchor=tk.E, pady=(0, 8))
    vista_camara = tk.Label(
        contenido,
        text="Iniciando camara y modelo...",
        font=("Segoe UI", 12),
        fg="#d1d5db",
        bg="#030712",
        width=80,
        height=30,
    )
    vista_camara.pack(fill=tk.BOTH, expand=True)

    hilo_camara = threading.Thread(
        target=procesar_camara,
        args=(
            cola_fotogramas,
            cola_errores,
            detener,
            estado_umbral,
            bloqueo_umbral,
            estado_camara,
            bloqueo_camara,
            cola_camaras,
        ),
        daemon=True,
    )

    def actualizar_interfaz():
        try:
            indices_camaras = cola_camaras.get_nowait()
        except queue.Empty:
            pass
        else:
            indices_por_nombre.clear()
            indices_por_nombre.update(
                {
                    f"Camara {indice}": indice
                    for indice in indices_camaras
                }
            )
            opciones = list(indices_por_nombre)
            selector_camara.configure(values=opciones)
            if opciones:
                with bloqueo_camara:
                    indice_actual = estado_camara["indice"]
                seleccion_camara.set(
                    next(
                        (
                            nombre
                            for nombre, indice in indices_por_nombre.items()
                            if indice == indice_actual
                        ),
                        opciones[0],
                    )
                )
                selector_camara.configure(state="readonly")
            else:
                seleccion_camara.set("Sin camaras disponibles")

        try:
            error = cola_errores.get_nowait()
        except queue.Empty:
            error = None
        if error is not None:
            detener.set()
            messagebox.showerror("Error de deteccion", str(error), parent=raiz)
            raiz.destroy()
            return

        try:
            fotograma, fps, conteos = cola_fotogramas.get_nowait()
        except queue.Empty:
            pass
        else:
            fotograma_rgb = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
            imagen = ImageTk.PhotoImage(Image.fromarray(fotograma_rgb))
            vista_camara.configure(image=imagen, text="")
            vista_camara.image = imagen
            etiqueta_fps.set(f"FPS: {fps:.1f}")
            for etiqueta, variable in variables_conteo.items():
                variable.set(str(conteos[etiqueta]))

        raiz.after(15, actualizar_interfaz)

    def cerrar_aplicacion():
        detener.set()
        raiz.destroy()

    def manejar_tecla(evento):
        if evento.char in ("q", "Q"):
            cerrar_aplicacion()

    raiz.protocol("WM_DELETE_WINDOW", cerrar_aplicacion)
    raiz.bind_all("<KeyPress>", manejar_tecla)
    hilo_camara.start()
    raiz.after(15, actualizar_interfaz)
    raiz.mainloop()
