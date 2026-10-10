# Mechatronics

## Deteccion

La deteccion en tiempo real usa el modelo entrenado `Deteccion/best.pt`. Ejecuta
`IXTLI.bat` para crear `.venv` si hace falta, instalar las dependencias de
`requirements.txt` e iniciar la aplicacion. Durante la preparacion aparece una
ventana con una barra animada de progreso. Si `.venv` y las librerias ya estan
instaladas, las comprueba y abre la aplicacion sin reinstalarlas. Se necesita
Python 3 con Tkinter y conexion a internet para instalar las dependencias.

Tambien puedes iniciar la aplicacion manualmente desde la raiz del proyecto:

```powershell
.\.venv\Scripts\python.exe src\detection.py
```

El modelo debe incluir las cinco clases del proyecto: botella, celular, taza,
tijeras y desarmador. La inferencia filtra cualquier otra clase del modelo y no
la dibuja. El modelo no tiene una clase "desconocido": un objeto ajeno al
dataset todavia podria confundirse con una clase permitida.

La interfaz de Tkinter esta separada en `src/interface.py`; la captura y la
inferencia estan en `src/detection.py`. Los botones `−` y `+` ajustan el umbral
de confianza entre 5 % y 95 %, en pasos de 5 %; debajo puedes seleccionar entre
las camaras disponibles. Los conteos del panel corresponden a las detecciones
acumuladas de objetos unicos vistos durante la sesion; se conservan al cambiar
de camara. La captura y la inferencia se ejecutan en un hilo de trabajo y
envian el fotograma mas reciente a la interfaz, para que esta no procese una
cola de imagenes atrasadas. El logo de `src/img/logo.png` aparece en el panel y
como icono de la ventana y de la barra de tareas de Ixtli.

## Metricas del modelo

Desde la raiz del proyecto, ejecuta el evaluador con:

```powershell
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py
```

Por defecto evalua el split `valid` a `imgsz=320`, para aproximarse al tamano
usado por la aplicacion. Guarda precision, recall y mAP en JSON/CSV, ademas de
las graficas y la matriz de confusion de Ultralytics, dentro de una nueva
carpeta en `metricas\resultados`. Usa `--split test` para seleccionar el split
de prueba o `--dry-run` para revisar las etiquetas y la cobertura sin ejecutar
el modelo. El dataset actual no tiene evaluacion para todas las clases: revisa
la advertencia del script antes de presentar los resultados como metricas de
las cinco clases. El FPS aproximado del reporte mide inferencia del modelo,
no incluye camara, seguimiento ni interfaz, por lo que no equivale al FPS de
la aplicacion en vivo.