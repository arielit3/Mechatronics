import cv2

def main():
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("No se pudo abrir la camara.")
        return

    print("Camara iniciada. Presiona Q para salir.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("No se pudo capturar el fotograma.")
                break

            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Mechatronics - Camara", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()