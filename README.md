# Ixtli — Mechatronics

Aplicación de escritorio para detectar objetos en tiempo real con una cámara y
un modelo YOLO entrenado para las clases botella, celular, taza, tijeras y
desarmador.

## Requisitos del sistema

### Requisitos básicos para ejecutar

- Windows; el lanzador `IXTLI.bat` usa comandos y rutas específicos de Windows.
- Python 3 de 64 bits, accesible mediante `py -3` o `python`. Debe incluir
  Tcl/Tk para mostrar las ventanas y `pythonw.exe` para que el lanzador inicie
  la aplicación sin consola.
- Cámara compatible con OpenCV y el backend DirectShow.
- El archivo de pesos `Deteccion\best.pt` (aproximadamente 5.5 MB).
- Espacio disponible para crear `.venv` e instalar las dependencias indicadas
  en `requirements.txt`.
- Conexión a Internet durante la primera instalación, salvo que las
  dependencias ya estén disponibles en el equipo.
- Pantalla de al menos 980 × 600 píxeles, que es el tamaño mínimo configurado
  para la ventana.

### Hardware y rendimiento

La aplicación ejecuta la inferencia en CPU; actualmente no requiere ni utiliza
una GPU para detección. No se ha establecido un requisito mínimo formal de
procesador o memoria mediante pruebas en distintos equipos.

Como referencia, en desarrollo se ha usado un Intel Core i5-6300U de 2 núcleos
y 4 hilos, con PyTorch en CPU; en ese equipo se observó un rendimiento cercano
a 15 FPS en la aplicación. Es una referencia de una configuración concreta,
no una garantía ni un mínimo universal. Para un uso más fluido se recomienda
un procesador más reciente y al menos 8 GB de RAM; este valor es recomendado,
no un mínimo medido. El rendimiento también depende de la cámara, la carga del
sistema y la escena.

La carpeta `dataset/` solo es necesaria para ejecutar la evaluación de métricas
o para tareas de entrenamiento; no es necesaria para abrir la aplicación de
detección. El modelo `Deteccion\best.pt` sí es necesario para la aplicación.

## Inicio rápido

Desde la carpeta raíz del proyecto, abre o ejecuta:

```powershell
.\IXTLI.bat
```

El lanzador busca Python, prepara el entorno cuando hace falta e inicia la
aplicación. La primera ejecución puede tardar mientras se crea `.venv` y se
instalan las dependencias. Las ejecuciones siguientes normalmente reutilizan
ese entorno.

## Qué hace `IXTLI.bat`

El archivo por lotes cambia primero a la carpeta del proyecto para que sus
rutas funcionen aunque se haya iniciado desde otro directorio. Luego:

1. Si existen `.venv\Scripts\python.exe` y `pythonw.exe`, comprueba que se
   puedan importar OpenCV, Ultralytics y Pillow. Si la comprobación pasa, inicia
   directamente `src\detection.py` con `pythonw.exe` y termina.
2. Si el entorno local no está listo, busca Python mediante `py -3` y después
   mediante `python`.
3. Comprueba que exista `pythonw.exe` junto al intérprete y que ese Python
   incluya Tkinter. Si falta alguno, muestra un mensaje de error en la consola.
4. Inicia `installer.py` con `pythonw.exe`. Este muestra una ventana de
   progreso, crea `.venv` si no existe, instala `requirements.txt` y vuelve a
   comprobar los imports principales.
5. Cuando termina la preparación, `installer.py` inicia
   `.venv\Scripts\pythonw.exe src\detection.py`. La ventana de preparación se
   cierra; la aplicación queda abierta.

`pythonw.exe` inicia Python sin una consola asociada. Si necesitas ver
mensajes de depuración, ejecuta la aplicación manualmente desde PowerShell.

### Actualizar o reinstalar dependencias

El lanzador no compara las versiones instaladas con `requirements.txt` cuando
los imports principales ya funcionan. Si se agregaron dependencias o cambió el
archivo de requisitos, actualiza el entorno manualmente desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No uses `pip install yaml`: el paquete de distribución que proporciona el
módulo `yaml` se llama `PyYAML`. El archivo de requisitos ya usa ese nombre.

## Ejecución manual

Para ejecutar con la consola visible:

```powershell
.\.venv\Scripts\python.exe src\detection.py
```

Si aún no existe `.venv`, crea el entorno e instala las dependencias:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uso de la aplicación

- La ventana muestra la cámara y un panel lateral.
- Los botones `−` y `+` ajustan el umbral de confianza de 5 % a 95 %, en pasos
  de 5 %. El valor inicial es 35 %.
- Un umbral mayor descarta detecciones con confianza más baja; no aumenta por
  sí mismo la precisión real del modelo. Un umbral menor puede recuperar
  detecciones, pero también mostrar falsos positivos.
- El selector permite elegir entre las cámaras que se detecten al iniciar.
- Las etiquetas sobre los objetos muestran su clase, confianza e ID temporal
  del rastreador ByteTrack.
- Los números del panel reflejan cuántas detecciones de cada clase hay en el
  fotograma procesado más reciente. No son un total acumulado de objetos únicos
  que hayan pasado frente a la cámara.
- Pulsa `Q` o `q` para cerrar la ventana; también puedes usar el botón de cerrar
  de Windows.

## Cómo funciona la detección

`src\detection.py` se encarga de buscar y abrir la cámara, cargar
`Deteccion\best.pt` y procesar los fotogramas. Busca índices de cámara del 1 al
9 y luego el 0 usando el backend DirectShow de OpenCV. Redimensiona cada
fotograma a 640 × 480 y llama a YOLO con `imgsz=320`, `device="cpu"` y el umbral
actual. ByteTrack se ejecuta con persistencia para asignar IDs mientras los
objetos se siguen entre fotogramas.

`src\interface.py` contiene la ventana de Tkinter, sus controles y las
actualizaciones visuales. La cámara y la inferencia corren en un hilo de
trabajo y pasan el fotograma más reciente mediante una cola pequeña. Esto evita
que la interfaz procese una cola larga de fotogramas antiguos; no elimina el
costo computacional de la inferencia YOLO ni garantiza una frecuencia de FPS
específica. El valor FPS que muestra la aplicación se calcula en el ciclo de
captura, inferencia y dibujo; no es una medición aislada del tiempo de YOLO.

El modelo contiene cinco clases: `bottle`, `cell_phone`, `cup`, `scissors` y
`screwdriver`. El código solo dibuja esas clases. No hay una clase de
"desconocido", por lo que un objeto ajeno a las clases entrenadas puede
clasificarse erróneamente como una de ellas.

## Justificación de la técnica elegida

El sistema combina detección de objetos con YOLO11n y seguimiento de
identificadores con ByteTrack. Se eligió esta combinación porque la aplicación
necesita localizar y clasificar varios objetos en cada fotograma de una cámara
en vivo, y además mantener un ID temporal cuando estos aparecen en fotogramas
consecutivos.

- **YOLO11n para detección:** el modelo produce clases, niveles de confianza y
  cajas delimitadoras en una sola inferencia por fotograma. La variante `n`
  (nano) es la opción ligera de la familia YOLO11 y resulta adecuada para
  priorizar velocidad y menor demanda de recursos en un equipo que ejecuta la
  inferencia en CPU. Se prefirió una detección por fotograma frente a un
  clasificador de imagen completa porque se requiere distinguir la posición
  de cada objeto y admitir varias instancias en la misma imagen.
- **ByteTrack para seguimiento:** la detección identifica objetos en cada
  imagen, pero por sí sola no conserva su identidad entre imágenes. ByteTrack
  asocia detecciones entre fotogramas y proporciona IDs temporales, lo que
  permite visualizar el seguimiento. No convierte automáticamente esos IDs
  en un conteo fiable de objetos que cruzaron la cámara: para eso hace falta
  definir una zona o línea de cruce y controlar reapariciones y oclusiones.
- **Umbral de confianza ajustable:** permite elegir el compromiso entre
  descartar predicciones dudosas o aceptar más detecciones potenciales. Es un
  filtro de salida, no un reentrenamiento ni una mejora intrínseca del modelo;
  su valor adecuado debe determinarse con datos de validación.
- **OpenCV y Tkinter en tareas separadas:** OpenCV obtiene los fotogramas y
  realiza el preprocesamiento/dibujo; Tkinter presenta la interfaz. La
  inferencia corre en un hilo de trabajo y comunica el fotograma más reciente
  mediante una cola limitada, para evitar que la interfaz acumule imágenes
  antiguas y se vuelva poco responsiva.

Esta selección es un compromiso entre localización, clasificación,
seguimiento visual y ejecución local. No garantiza por sí sola una exactitud o
un FPS determinados. La aplicación usa CPU y procesa cada fotograma
secuencialmente; los datos de evaluación disponibles tampoco cubren todavía
las cinco clases. Por ello, cualquier conclusión de calidad o velocidad debe
respaldarse con métricas completas y pruebas en el hardware objetivo. La
sección de [limitaciones](#limitaciones-conocidas) y las
[mejoras propuestas](#mejoras-propuestas) describen los siguientes pasos.

## Métricas del modelo

El evaluador `metricas\evaluar_modelo.py` usa la validación de Ultralytics para
calcular precisión, recall, mAP@0.5 y mAP@0.5:0.95, generar gráficas y matriz de
confusión, y escribir un resumen JSON y una tabla CSV por clase.

Primero puedes revisar los datos y las etiquetas sin ejecutar inferencia:

```powershell
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py --dry-run
```

Para calcular las métricas con el modelo:

```powershell
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py
```

El evaluador usa `valid` e `imgsz=320` por defecto. Las salidas de cada
ejecución se guardan en una carpeta con fecha y hora dentro de
`metricas\resultados\`. Opciones disponibles:

```powershell
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py --help
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py --split test
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py --imgsz 640
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py --model "ruta\al\modelo.pt"
.\.venv\Scripts\python.exe metricas\evaluar_modelo.py --dataset "ruta\al\dataset"
```

El FPS aproximado que se calcula en el reporte deriva del tiempo de inferencia
de YOLO. No incluye la cámara, ByteTrack, el dibujo ni Tkinter y, por tanto, no
se debe comparar directamente con los FPS de la aplicación en vivo.

### Cobertura y validez de las métricas actuales

El evaluador combina los directorios por clase de `dataset\` y remapea sus
etiquetas locales (clase `0` dentro de cada carpeta) al ID correspondiente del
modelo. Las imágenes no se modifican; se enlazan o copian a un conjunto
temporal que se elimina al terminar.

En la estructura de datos presente en el proyecto:

- `valid` incluye botella, celular, taza y desarmador, pero no tijeras.
- `test` incluye botella, celular y taza; no incluye desarmador ni tijeras.

Por ello, los resultados de esos splits no representan una evaluación
completa de las cinco clases. El script muestra una advertencia sobre las
clases sin datos. Para un reporte final comparable, hace falta un split de
evaluación etiquetado con ejemplos representativos para cada clase, sin
reutilizar imágenes de entrenamiento.

## Estructura relevante

```text
IXTLI.bat                 Lanzador de Windows
installer.py              Preparación de .venv e instalación de dependencias
requirements.txt          Dependencias de Python
src/
  detection.py            Cámara, modelo, inferencia y ByteTrack
  interface.py            Ventana y controles Tkinter
  img/logo.png            Logo de la aplicación
Deteccion/
  best.pt                 Pesos del modelo usado por la aplicación
metricas/
  evaluar_modelo.py       Evaluador y generador de reportes
  resultados/             Resultados de evaluación (se crean al evaluar)
dataset/                  Imágenes y etiquetas organizadas por clase
```

El entorno `.venv/` y `dataset/` están excluidos de Git. Según cómo se comparta
el proyecto, puede ser necesario obtenerlos o prepararlos localmente. El
modelo `Deteccion\best.pt` también debe estar presente para iniciar la
aplicación y evaluar.

## Limitaciones conocidas

- **Rendimiento:** la inferencia está fijada a CPU y procesa una imagen por
  iteración. El rendimiento depende del procesador, la cámara y la carga del
  sistema; no se promete un FPS mínimo. En equipos modestos puede ser bajo.
- **Uso de GPU:** el código de la aplicación especifica CPU. ONNX, OpenVINO o
  aceleración por GPU no están integrados ni validados actualmente.
- **Cobertura de datos:** faltan imágenes de evaluación para tijeras en `valid`
  y tanto para desarmador como para tijeras en `test`.
- **Objetos desconocidos:** el modelo no fue definido para rechazar todo
  objeto que no pertenezca a las cinco clases.
- **Conteo:** el panel informa detecciones del último fotograma, no entradas,
  salidas ni un historial persistente de objetos únicos. Los IDs de ByteTrack
  son temporales y no equivalen a una identidad física permanente.
- **Cámaras:** la búsqueda está limitada a índices 0–9 y usa DirectShow; otros
  backends, cámaras de red o índices superiores no están configurados.
- **Entorno/instalación:** se necesita Python con Tkinter y `pythonw.exe`.
  Instalar dependencias requiere Internet si no están disponibles localmente.
  El lanzador solo comprueba algunos imports principales antes de omitir la
  instalación de requisitos.
- **Dataset distribuido:** la carpeta `dataset/` está ignorada por Git, por lo
  que no necesariamente viene incluida al clonar el repositorio.

## Mejoras propuestas

1. **Completar y revisar el dataset:** incorporar validación y prueba para todas
   las clases, revisar calidad de etiquetas y evitar que imágenes repetidas o
   casi idénticas aparezcan en entrenamiento y evaluación.
2. **Medir rendimiento por etapa:** registrar por separado captura,
   preprocesamiento, YOLO, ByteTrack, dibujo y visualización usando la misma
   cámara y condiciones de prueba.
3. **Optimizar inferencia con evidencia:** comparar PyTorch con ONNX Runtime o
   OpenVINO y revisar cuantización, tamaño de entrada y modelo. Aceptar una
   optimización solo si mejora el FPS real y mantiene métricas de detección
   suficientes.
4. **Conteo de paso:** definir una línea o zona de cruce, contar cada ID una
   sola vez y distinguir entradas/salidas; documentar los límites del
   seguimiento cuando haya oclusiones.
5. **Rechazo de desconocidos:** recolectar ejemplos negativos y diseñar una
   estrategia de rechazo, validándola con falsos positivos y falsos negativos.
6. **Instalación reproducible:** fijar versiones probadas de dependencias,
   comprobar todos los imports requeridos y mostrar errores útiles cuando no
   haya conexión o falle la instalación.
7. **Pruebas automatizadas:** añadir pruebas para configuración, manejo de
   etiquetas, umbral, selección de cámara y generación de reportes.
