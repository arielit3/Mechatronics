
import cv2
import time
import colorsys
import queue
from pathlib import Path
from ultralytics import YOLO

MAX_INDICE_CAMARA = 9
TAMANO_FOTOGRAMA = (640, 480)

# Nombres de clase que contiene el modelo entrenado.
CLASES_PERMITIDAS = {
    "bottle": "Botella",
    "cell_phone": "Celular",
    "cup": "Taza",
    "scissors": "Tijeras",
    "screwdriver": "Destornillador"
}
RUTA_MODELO = Path(__file__).resolve().parents[1] / "Deteccion" / "best.pt"


def color_por_id(id_objeto):
    tono = (id_objeto * 0.61803398875) % 1.0
    rojo, verde, azul = colorsys.hsv_to_rgb(tono, 0.85, 1.0)
    return int(azul * 255), int(verde * 255), int(rojo * 255)


def buscar_camara():
    # Las camaras USB adicionales suelen tener indices mayores que 0;
    # se prueban primero y la camara integrada queda como alternativa.
    indices = [*range(1, MAX_INDICE_CAMARA + 1), 0]
    for indice in indices:
        camara = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
        if camara.isOpened():
            correcto, primer_fotograma = camara.read()
            if correcto:
                return camara, indice, primer_fotograma
        camara.release()

    return None, None, None


def publicar_fotograma(cola_fotogramas, datos):
    try:
        cola_fotogramas.put_nowait(datos)
    except queue.Full:
        try:
            cola_fotogramas.get_nowait()
        except queue.Empty:
            pass
        cola_fotogramas.put_nowait(datos)


def procesar_camara(
    cola_fotogramas,
    cola_errores,
    detener,
    estado_umbral,
    bloqueo_umbral,
):
    camara = None
    try:
        camara, _, primer_fotograma = buscar_camara()
        if camara is None:
            raise RuntimeError("No se encontro ninguna camara disponible.")

        if not RUTA_MODELO.is_file():
            raise FileNotFoundError(f"No se encontro el modelo entrenado: {RUTA_MODELO}")

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

        tiempo_anterior = time.perf_counter()
        while not detener.is_set():
            if primer_fotograma is not None:
                fotograma = primer_fotograma
                primer_fotograma = None
                correcto = True
            else:
                correcto, fotograma = camara.read()

            if not correcto:
                raise RuntimeError("Error al capturar un fotograma de la camara.")

            fotograma = cv2.resize(fotograma, TAMANO_FOTOGRAMA)
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
                verbose=False
            )

            imagen = fotograma.copy()
            conteos = {etiqueta: 0 for etiqueta in CLASES_PERMITIDAS.values()}

            for resultado in resultados:
                for indice_caja, caja in enumerate(resultado.boxes):
                    id_clase = int(caja.cls[0])
                    nombre = modelo.names[id_clase]
                    id_objeto = (
                        int(caja.id[0])
                        if caja.id is not None
                        else None
                    )

                    x1, y1, x2, y2 = map(
                        int, caja.xyxy[0].tolist()
                    )

                    confianza = float(caja.conf[0])
                    etiqueta = CLASES_PERMITIDAS.get(nombre)
                    if etiqueta is None:
                        continue
                    conteos[etiqueta] += 1
                    id_color = (
                        id_objeto
                        if id_objeto is not None
                        else id_clase + indice_caja
                    )
                    color = color_por_id(id_color)

                    cv2.rectangle(
                        imagen,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        imagen,
                        f"{etiqueta} ID:{id_objeto if id_objeto is not None else '--'} "
                        f"{confianza:.0%}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            tiempo_actual = time.perf_counter()
            tiempo_transcurrido = tiempo_actual - tiempo_anterior
            fps = 1.0 / tiempo_transcurrido if tiempo_transcurrido > 0 else 0.0
            tiempo_anterior = tiempo_actual

            cv2.putText(
                imagen,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            publicar_fotograma(cola_fotogramas, (imagen, fps, conteos))
    except Exception as error:
        cola_errores.put(error)
    finally:
        if camara is not None:
            camara.release()


if __name__ == "__main__":
    if __package__:
        from .interface import ejecutar_interfaz
    else:
        from interface import ejecutar_interfaz

    ejecutar_interfaz(procesar_camara, CLASES_PERMITIDAS)
