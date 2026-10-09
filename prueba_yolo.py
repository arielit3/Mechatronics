
from ultralytics import YOLO

# Cargar un modelo pequeño preentrenado
modelo = YOLO("yolo11n.pt")

# Detectar objetos en una imagen de prueba
resultados = modelo.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device="cpu",
    imgsz=320,
    save=True
)

print("Prueba terminada.")
print("Imagen con detecciones guardada en runs/detect/predict/")
