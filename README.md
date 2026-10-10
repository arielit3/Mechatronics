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

La interfaz de Tkinter esta separada en `src/interface.py`; la captura y la
inferencia estan en `src/detection.py`. Los botones `−` y `+` ajustan el umbral
de confianza entre 5 % y 95 %, en pasos de 5 %. Los conteos del panel
corresponden a las detecciones del fotograma visible; no son un acumulado de
objetos que hayan pasado por la camara. La captura y la inferencia se ejecutan
en un hilo de trabajo y envian el fotograma mas reciente a la interfaz, para
que esta no procese una cola de imagenes atrasadas.