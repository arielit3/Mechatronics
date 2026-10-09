
import cv2
from ultralytics import YOLO
import time


def main():
    # Cargar el modelo YOLO pequeño
    modelo = YOLO("yolo11n.pt")

    # Abrir la webcam
    camara = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not camara.isOpened():
        print("Error: no se pudo abrir la camara.")
        return

    print("Mechatronics iniciado. Presiona Q para salir.")

    tiempo_anterior = time.time()

    try:
        while True:
            correcto, fotograma = camara.read()

            if not correcto:
                print("Error al capturar el fotograma.")
                break

            # Reducir resolucion para trabajar mejor con CPU
            fotograma = cv2.resize(fotograma, (640, 480))

            # Detectar objetos usando CPU
            resultados = modelo.predict(
                source=fotograma,
                device="cpu",
                imgsz=320,
                conf=0.35,
                verbose=False
            )

            # Dibujar las detecciones
            imagen = resultados[0].plot()

            # Calcular FPS aproximados
            tiempo_actual = time.time()
            diferencia = tiempo_actual - tiempo_anterior
            fps = 1 / diferencia if diferencia > 0 else 0
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

            cv2.imshow("Mechatronics - Deteccion en tiempo real", imagen)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
