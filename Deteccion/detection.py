from pathlib import Path

import cv2
from ultralytics import YOLO

CLASES_PERMITIDAS = {
    "bottle",
    "cell_phone",
    "cup",
    "scissors",
    "screwdriver",
}
RUTA_MODELO = Path(__file__).resolve().with_name("best.pt")

if not RUTA_MODELO.is_file():
    raise FileNotFoundError(f"No se encontro el modelo entrenado: {RUTA_MODELO}")

model = YOLO(str(RUTA_MODELO))
clases_faltantes = CLASES_PERMITIDAS - set(model.names.values())
if clases_faltantes:
    raise RuntimeError(
        "El modelo no contiene todas las clases esperadas: "
        + ", ".join(sorted(clases_faltantes))
    )
ids_clases_permitidas = [
    class_id
    for class_id, name in model.names.items()
    if name in CLASES_PERMITIDAS
]

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("No se pudo abrir la cámara. Prueba con 0 en lugar de 1.")
    raise SystemExit

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer la imagen de la cámara.")
            break

        results = model.predict(
            frame,
            device="cpu",
            conf=0.25,
            imgsz=640,
            classes=ids_clases_permitidas,
            verbose=False,
        )
        annotated = results[0].plot()

        cv2.imshow("Deteccion de objetos (Q para salir)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()