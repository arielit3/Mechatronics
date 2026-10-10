# Mechatronics

## Deteccion

La deteccion en tiempo real usa el modelo entrenado `Deteccion/best.pt`. Desde la
raiz del proyecto, inicia la aplicacion con:

```powershell
.\.venv\Scripts\python.exe src\detection.py
```

El modelo debe incluir las cinco clases del proyecto: botella, celular, taza,
tijeras y desarmador. La inferencia filtra cualquier otra clase del modelo y no
la dibuja. El modelo no tiene una clase "desconocido": un objeto ajeno al
dataset todavia podria confundirse con una clase permitida.