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