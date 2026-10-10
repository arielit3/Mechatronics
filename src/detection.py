
import ctypes
import cv2
import time
import colorsys
from pathlib import Path
from ultralytics import YOLO

MAX_INDICE_CAMARA = 9

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


def mostrar_error_camara():
    ctypes.windll.user32.MessageBoxW(
        None,
        "No se encontro ninguna camara disponible. Conecta una camara e "
        "intenta ejecutar el programa nuevamente.",
        "Mechatronics - Camara no disponible",
        0x10,
    )


def main():
    camara, indice_camara, primer_fotograma = buscar_camara()
    if camara is None:
        mostrar_error_camara()
        return

    print(f"Camara {indice_camara} detectada.")

    try:
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

        print("Mechatronics iniciado. Presiona Q para salir.")

        # Variables para calcular FPS
        tiempo_anterior = time.perf_counter()
        fps = 0.0

        while True:
            if primer_fotograma is not None:
                fotograma = primer_fotograma
                primer_fotograma = None
                correcto = True
            else:
                correcto, fotograma = camara.read()

            if not correcto:
                print("Error al capturar el fotograma.")
                break

            # Reducir resolucion para mejorar el rendimiento
            fotograma = cv2.resize(fotograma, (640, 480))

            # Detectar objetos usando exclusivamente CPU
            resultados = modelo.track(
                source=fotograma,
                device="cpu",
                imgsz=320,
                conf=0.35,
                classes=ids_clases_permitidas,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            # Copiar imagen para dibujar las detecciones
            imagen = fotograma.copy()

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
                    id_color = (
                        id_objeto
                        if id_objeto is not None
                        else id_clase + indice_caja
                    )
                    color = color_por_id(id_color)

                    # Dibujar rectangulo
                    cv2.rectangle(
                        imagen,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    # Dibujar nombre y confianza
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

            # Calcular FPS del ciclo completo
            tiempo_actual = time.perf_counter()
            tiempo_transcurrido = tiempo_actual - tiempo_anterior

            if tiempo_transcurrido > 0:
                fps = 1.0 / tiempo_transcurrido

            tiempo_anterior = tiempo_actual

            # Mostrar contador de FPS
            cv2.putText(
                imagen,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # INTERFAZ: agrega aqui elementos visuales como botones dibujados,
            # estados o instrucciones antes de presentar cada fotograma.
            cv2.imshow(
                "Ixtli - Deteccion en tiempo real",
                imagen
            )

            # INTERFAZ: para responder a clics, registra cv2.setMouseCallback
            # en la ventana y procesa aqui las acciones junto con las teclas.
            # OpenCV no ofrece botones nativos; para controles reales, integra
            # esta vista en una interfaz hecha con Tkinter o PySide.
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q")):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
