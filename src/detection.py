
import cv2
import time
from ultralytics import YOLO

# Clases que queremos reconocer
CLASES_PERMITIDAS = {
    "cup": "Taza",
    "scissors": "Tijeras",
    "bottle": "Botella",
    "cell phone": "Celular",
    "screwdriver": "Destornillador"
}


def main():
    # Cargar el modelo YOLO
    modelo = YOLO("yolo11n.pt")

    # Seleccionar camara: 0 para la primera, 1 para la segunda
    indice_camara = 0
    camara = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)

    if not camara.isOpened():
        print(f"Error: no se pudo abrir la camara {indice_camara}.")
        return

    print("Mechatronics iniciado. Presiona Q para salir.")

    # Variables para calcular FPS
    tiempo_anterior = time.perf_counter()
    fps = 0.0

    try:
        while True:
            correcto, fotograma = camara.read()

            if not correcto:
                print("Error al capturar el fotograma.")
                break

            # Reducir resolucion para mejorar el rendimiento
            fotograma = cv2.resize(fotograma, (640, 480))

            # Detectar objetos usando exclusivamente CPU
            resultados = modelo.predict(
                source=fotograma,
                device="cpu",
                imgsz=320,
                conf=0.35,
                verbose=False
            )

            # Copiar imagen para dibujar las detecciones
            imagen = fotograma.copy()

            for resultado in resultados:
                for caja in resultado.boxes:
                    id_clase = int(caja.cls[0])
                    nombre = modelo.names[id_clase]

                    x1, y1, x2, y2 = map(
                        int, caja.xyxy[0].tolist()
                    )

                    confianza = float(caja.conf[0])

                    if nombre in CLASES_PERMITIDAS:
                        etiqueta = CLASES_PERMITIDAS[nombre]
                        color = (0, 255, 0)  # Verde
                    else:
                        etiqueta = "Objeto desconocido"
                        color = (0, 0, 255)  # Rojo

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
                        f"{etiqueta} {confianza:.0%}",
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

            # Mostrar ventana
            cv2.imshow(
                "Mechatronics - Deteccion en tiempo real",
                imagen
            )

            # Salir al presionar Q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
