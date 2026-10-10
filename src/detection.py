import colorsys
import queue
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from interface import ejecutar_interfaz


MAX_INDICE_CAMARA = 9

# Nombres de clase que contiene el modelo entrenado.
CLASES_PERMITIDAS = {
    "bottle": "Botella",
    "cell_phone": "Celular",
    "cup": "Taza",
    "scissors": "Tijeras",
    "screwdriver": "Destornillador",
}
RUTA_MODELO = Path(__file__).resolve().parents[1] / "Deteccion" / "best.pt"


def color_por_id(id_objeto):
    tono = (id_objeto * 0.61803398875) % 1.0
    rojo, verde, azul = colorsys.hsv_to_rgb(tono, 0.85, 1.0)
    return int(azul * 255), int(verde * 255), int(rojo * 255)


def buscar_camaras():
    indices = [*range(1, MAX_INDICE_CAMARA + 1), 0]
    disponibles = []
    for indice in indices:
        camara = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        try:
            if camara.isOpened():
                correcto, _ = camara.read()
                if correcto:
                    disponibles.append(indice)
        finally:
            camara.release()
    return disponibles


def procesar_camara(
    cola_fotogramas,
    cola_errores,
    detener,
    estado_umbral,
    bloqueo_umbral,
    estado_camara,
    bloqueo_camara,
    cola_camaras,
):
    camara = None
    try:
        indices_camaras = buscar_camaras()
        cola_camaras.put(indices_camaras)
        if not indices_camaras:
            raise RuntimeError(
                "No se encontro ninguna camara disponible. "
                "Conecta una camara e intenta nuevamente."
            )

        with bloqueo_camara:
            indice_camara = estado_camara["indice"]
            if indice_camara not in indices_camaras:
                indice_camara = indices_camaras[0]
                estado_camara["indice"] = indice_camara

        camara, primer_fotograma = abrir_camara(indice_camara)
        if camara is None:
            raise RuntimeError(f"No se pudo abrir la camara {indice_camara}.")

        if not RUTA_MODELO.is_file():
            raise FileNotFoundError(
                f"No se encontro el modelo entrenado: {RUTA_MODELO}"
            )

        modelo = YOLO(str(RUTA_MODELO))
        clases_modelo = modelo.names
        clases_faltantes = set(CLASES_PERMITIDAS) - set(clases_modelo.values())
        if clases_faltantes:
            raise RuntimeError(
                "El modelo no contiene todas las clases esperadas: "
                + ", ".join(sorted(clases_faltantes))
            )
        ids_clases_permitidas = [
            id_clase
            for id_clase, nombre in clases_modelo.items()
            if nombre in CLASES_PERMITIDAS
        ]

        print(f"Camara {indice_camara} detectada.")
        inicio_fps = time.perf_counter()
        fotogramas_por_segundo = 0
        conteos_acumulados = {
            etiqueta: 0 for etiqueta in CLASES_PERMITIDAS.values()
        }
        objetos_vistos = set()

        while not detener.is_set():
            with bloqueo_camara:
                indice_solicitado = estado_camara["indice"]
            if indice_solicitado != indice_camara:
                camara.release()
                camara, primer_fotograma = abrir_camara(indice_solicitado)
                if camara is None:
                    raise RuntimeError(
                        f"No se pudo cambiar a la camara {indice_solicitado}."
                    )
                indice_camara = indice_solicitado
                inicio_fps = time.perf_counter()
                fotogramas_por_segundo = 0

            if primer_fotograma is not None:
                fotograma = primer_fotograma
                primer_fotograma = None
                correcto = True
            else:
                correcto, fotograma = camara.read()

            if not correcto:
                raise RuntimeError("Error al capturar el fotograma.")

            fotograma = cv2.resize(fotograma, (640, 480))
            with bloqueo_umbral:
                umbral = estado_umbral["valor"]

            resultados = modelo.track(
                source=fotograma,
                device="cpu",
                imgsz=320,
                conf=umbral,
                classes=ids_clases_permitidas,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            imagen = fotograma.copy()
            imagen = fotograma.copy()
            for resultado in resultados:
                for indice_caja, caja in enumerate(resultado.boxes):
                    id_clase = int(caja.cls[0])
                    nombre = modelo.names[id_clase]
                    etiqueta = CLASES_PERMITIDAS.get(nombre)
                    if etiqueta is None:
                        continue

                    id_objeto = (
                        int(caja.id[0])
                        if caja.id is not None
                        else None
                    )
                    clave_objeto = (
                        (indice_camara, id_objeto)
                        if id_objeto is not None
                        else None
                    )
                    if clave_objeto is not None and clave_objeto not in objetos_vistos:
                        objetos_vistos.add(clave_objeto)
                        conteos_acumulados[etiqueta] += 1

                    x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
                    confianza = float(caja.conf[0])
                    id_color = (
                        id_objeto
                        if id_objeto is not None
                        else id_clase + indice_caja
                    )
                    color = color_por_id(id_color)

                    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        imagen,
                        f"{etiqueta} ID:{id_objeto if id_objeto is not None else '--'} "
                        f"{confianza:.0%}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            fotogramas_por_segundo += 1
            ahora = time.perf_counter()
            lapso = ahora - inicio_fps
            if lapso >= 1:
                fps = fotogramas_por_segundo / lapso
                inicio_fps = ahora
                fotogramas_por_segundo = 0
            else:
                fps = fotogramas_por_segundo / lapso

            try:
                cola_fotogramas.put_nowait(
                    (imagen, fps, conteos_acumulados.copy())
                )
            except queue.Full:
                try:
                    cola_fotogramas.get_nowait()
                except queue.Empty:
                    pass
                cola_fotogramas.put_nowait(
                    (imagen, fps, conteos_acumulados.copy())
                )
    except Exception as error:
        cola_errores.put(error)
    finally:
        if camara is not None:
            camara.release()


def abrir_camara(indice):
    camara = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if camara.isOpened():
        correcto, primer_fotograma = camara.read()
        if correcto:
            return camara, primer_fotograma
    camara.release()
    return None, None


def main():
    ejecutar_interfaz(procesar_camara, CLASES_PERMITIDAS)


if __name__ == "__main__":
    main()
