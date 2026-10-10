"""Evalua el modelo YOLO y guarda metricas y graficas en metricas/resultados."""

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIRECTORIO_METRICAS = Path(__file__).resolve().parent
MODELO_PREDETERMINADO = RAIZ_PROYECTO / "Deteccion" / "best.pt"
DATASET_PREDETERMINADO = RAIZ_PROYECTO / "dataset"

CARPETAS_POR_CLASE = {
    "Botella": "bottle",
    "Celular": "cell_phone",
    "Desarmador": "screwdriver",
    "Taza": "cup",
    "Tijeras": "scissors",
}
EXTENSIONES_IMAGEN = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def obtener_nombres_clase(nombres_modelo):
    if isinstance(nombres_modelo, dict):
        return [nombres_modelo[indice] for indice in sorted(nombres_modelo)]
    return list(nombres_modelo)


def cargar_etiquetas(ruta_etiqueta, id_clase_modelo):
    lineas_salida = []
    objetos = 0

    for numero_linea, linea in enumerate(
        ruta_etiqueta.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        partes = linea.split()
        if not partes:
            continue
        if len(partes) != 5:
            raise ValueError(
                f"Formato YOLO invalido en {ruta_etiqueta}, linea {numero_linea}."
            )
        if partes[0] != "0":
            raise ValueError(
                f"Se esperaba la clase local 0 en {ruta_etiqueta}, "
                f"linea {numero_linea}; se encontro {partes[0]}."
            )

        coordenadas = [float(valor) for valor in partes[1:]]
        if not all(math.isfinite(valor) and 0 <= valor <= 1 for valor in coordenadas):
            raise ValueError(
                f"Coordenadas fuera del rango [0, 1] en "
                f"{ruta_etiqueta}, linea {numero_linea}."
            )
        if coordenadas[2] <= 0 or coordenadas[3] <= 0:
            raise ValueError(
                f"Ancho y alto deben ser positivos en "
                f"{ruta_etiqueta}, linea {numero_linea}."
            )

        lineas_salida.append(
            f"{id_clase_modelo} " + " ".join(partes[1:])
        )
        objetos += 1

    return lineas_salida, objetos


def enlazar_o_copiar(origen, destino):
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def preparar_datos(dataset_root, split, nombres_modelo, directorio_temporal):
    ids_modelo_por_nombre = {
        nombre: indice for indice, nombre in enumerate(nombres_modelo)
    }
    imagenes_destino = directorio_temporal / "images" / "val"
    etiquetas_destino = directorio_temporal / "labels" / "val"
    imagenes_destino.mkdir(parents=True)
    etiquetas_destino.mkdir(parents=True)

    conteos = {
        nombre_modelo: {"imagenes": 0, "objetos": 0}
        for nombre_modelo in nombres_modelo
    }
    clases_ausentes = []
    total_imagenes = 0

    for carpeta_clase, nombre_modelo in CARPETAS_POR_CLASE.items():
        if nombre_modelo not in ids_modelo_por_nombre:
            raise ValueError(
                f"El modelo no contiene la clase requerida '{nombre_modelo}'."
            )

        origen_imagenes = dataset_root / carpeta_clase / split / "images"
        origen_etiquetas = dataset_root / carpeta_clase / split / "labels"
        if not origen_imagenes.is_dir() or not origen_etiquetas.is_dir():
            clases_ausentes.append(nombre_modelo)
            continue

        imagenes = sorted(
            ruta
            for ruta in origen_imagenes.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
        )
        etiquetas = {
            ruta.stem: ruta
            for ruta in origen_etiquetas.glob("*.txt")
            if ruta.is_file()
        }
        imagenes_por_nombre = {ruta.stem: ruta for ruta in imagenes}

        if not imagenes:
            clases_ausentes.append(nombre_modelo)
            continue

        etiquetas_sin_imagen = etiquetas.keys() - imagenes_por_nombre.keys()
        if etiquetas_sin_imagen:
            ejemplo = sorted(etiquetas_sin_imagen)[0]
            raise ValueError(
                f"Hay etiqueta sin imagen para {carpeta_clase}: {ejemplo}. "
                f"Revisa el split '{split}'."
            )

        id_clase = ids_modelo_por_nombre[nombre_modelo]
        for imagen in imagenes:
            etiqueta = etiquetas.get(imagen.stem)
            if etiqueta is None:
                raise ValueError(
                    f"Falta la etiqueta YOLO correspondiente a {imagen}."
                )

            nombre_archivo = f"{carpeta_clase}__{imagen.name}"
            enlazar_o_copiar(imagen, imagenes_destino / nombre_archivo)
            lineas, cantidad_objetos = cargar_etiquetas(etiqueta, id_clase)
            (etiquetas_destino / f"{carpeta_clase}__{imagen.stem}.txt").write_text(
                "\n".join(lineas) + ("\n" if lineas else ""),
                encoding="utf-8",
            )
            conteos[nombre_modelo]["imagenes"] += 1
            conteos[nombre_modelo]["objetos"] += cantidad_objetos
            total_imagenes += 1

    if total_imagenes == 0:
        raise FileNotFoundError(
            f"No se encontraron imagenes etiquetadas para el split '{split}' "
            f"en {dataset_root}."
        )

    clases_sin_objetos = [
        nombre
        for nombre, conteo in conteos.items()
        if conteo["imagenes"] and conteo["objetos"] == 0
    ]
    for nombre in clases_sin_objetos:
        if nombre not in clases_ausentes:
            clases_ausentes.append(nombre)

    configuracion = {
        "path": directorio_temporal.as_posix(),
        # Ultralytics requires train in the YAML; model.val only reads val.
        "train": "images/val",
        "val": "images/val",
        "nc": len(nombres_modelo),
        "names": nombres_modelo,
    }
    ruta_yaml = directorio_temporal / "data.yaml"
    ruta_yaml.write_text(
        yaml.safe_dump(configuracion, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return ruta_yaml, conteos, clases_ausentes, total_imagenes


def guardar_metricas(metricas, conteos, nombres_modelo, resumen_path, clases_path):
    metricas_por_id = {}
    for indice, id_clase in enumerate(metricas.box.ap_class_index):
        precision, recall, map50, map50_95 = metricas.box.class_result(indice)
        metricas_por_id[int(id_clase)] = {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }

    filas = []
    for id_clase, nombre in enumerate(nombres_modelo):
        soporte = conteos[nombre]["objetos"]
        valores = metricas_por_id.get(id_clase, {})
        filas.append(
            {
                "clase": nombre,
                "imagenes": conteos[nombre]["imagenes"],
                "objetos_reales": soporte,
                "precision": valores.get("precision"),
                "recall": valores.get("recall"),
                "mAP50": valores.get("map50"),
                "mAP50_95": valores.get("map50_95"),
            }
        )

    with clases_path.open("w", newline="", encoding="utf-8-sig") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=filas[0].keys())
        escritor.writeheader()
        escritor.writerows(filas)

    velocidad = getattr(metricas, "speed", {}) or {}
    ms_inferencia = velocidad.get("inference")
    resumen = {
        "precision_media": float(metricas.box.mp),
        "recall_medio": float(metricas.box.mr),
        "mAP50": float(metricas.box.map50),
        "mAP50_95": float(metricas.box.map),
        "velocidad_ms_por_imagen": {
            nombre: float(valor) for nombre, valor in velocidad.items()
        },
        "fps_inferencia_aproximado": (
            1000 / ms_inferencia if ms_inferencia and ms_inferencia > 0 else None
        ),
        "metricas_por_clase_csv": clases_path.name,
    }
    resumen_path.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return resumen


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calcula precision, recall, mAP y graficas de evaluacion YOLO. "
            "Combina las carpetas por clase del dataset en un conjunto temporal."
        )
    )
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="valid",
        help="Particion a evaluar; por defecto usa valid.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Tamano de inferencia; 320 coincide con la aplicacion actual.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELO_PREDETERMINADO,
        help="Ruta al modelo .pt.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PREDETERMINADO,
        help="Carpeta que contiene Botella, Celular, etc.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Dispositivo de Ultralytics; por defecto cpu.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida las etiquetas y muestra cobertura sin ejecutar inferencia.",
    )
    argumentos = parser.parse_args()

    modelo_path = argumentos.model.resolve()
    dataset_root = argumentos.dataset.resolve()
    if not modelo_path.is_file():
        raise FileNotFoundError(f"No se encontro el modelo: {modelo_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"No se encontro el dataset: {dataset_root}")
    if argumentos.imgsz <= 0:
        parser.error("--imgsz debe ser un entero positivo.")

    DIRECTORIO_METRICAS.mkdir(parents=True, exist_ok=True)
    modelo = YOLO(str(modelo_path))
    nombres_modelo = obtener_nombres_clase(modelo.names)
    nombre_ejecucion = (
        f"{argumentos.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    directorio_salida = DIRECTORIO_METRICAS / "resultados" / nombre_ejecucion

    with tempfile.TemporaryDirectory(
        prefix=".dataset_temporal_",
        dir=DIRECTORIO_METRICAS,
    ) as temporal:
        ruta_yaml, conteos, clases_ausentes, total_imagenes = preparar_datos(
            dataset_root,
            argumentos.split,
            nombres_modelo,
            Path(temporal),
        )

        print(f"Modelo: {modelo_path}")
        print(f"Split: {argumentos.split}")
        print(f"Imagenes evaluables: {total_imagenes}")
        for nombre, conteo in conteos.items():
            if conteo["imagenes"]:
                print(
                    f"  {nombre}: {conteo['imagenes']} imagenes, "
                    f"{conteo['objetos']} objetos"
                )
        if clases_ausentes:
            print(
                "ADVERTENCIA: no hay datos de evaluacion para: "
                + ", ".join(clases_ausentes)
            )

        if argumentos.dry_run:
            print("Validacion de datos completada; no se ejecuto la inferencia.")
            return

        directorio_salida.mkdir(parents=True, exist_ok=False)
        metricas = modelo.val(
            data=str(ruta_yaml),
            imgsz=argumentos.imgsz,
            device=argumentos.device,
            workers=0,
            plots=True,
            project=str(directorio_salida.parent),
            name=directorio_salida.name,
            exist_ok=True,
        )

    resumen_path = directorio_salida / "resumen_metricas.json"
    clases_path = directorio_salida / "metricas_por_clase.csv"
    resumen = guardar_metricas(
        metricas,
        conteos,
        nombres_modelo,
        resumen_path,
        clases_path,
    )

    resumen["split"] = argumentos.split
    resumen["tamano_entrada"] = argumentos.imgsz
    resumen["dispositivo"] = argumentos.device
    resumen["clases_sin_datos"] = clases_ausentes
    resumen_path.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nMetricas globales:")
    print(f"  Precision: {resumen['precision_media']:.4f}")
    print(f"  Recall:    {resumen['recall_medio']:.4f}")
    print(f"  mAP50:     {resumen['mAP50']:.4f}")
    print(f"  mAP50-95:  {resumen['mAP50_95']:.4f}")
    if resumen["fps_inferencia_aproximado"] is not None:
        print(
            "  FPS inferencia aproximado: "
            f"{resumen['fps_inferencia_aproximado']:.2f}"
        )
    print(f"\nResultados guardados en: {directorio_salida}")
    print(f"  {resumen_path.name}")
    print(f"  {clases_path.name}")
    print("  Graficas y matriz de confusion de Ultralytics")
    if clases_ausentes:
        print(
            "\nEstas metricas no cubren todas las clases del modelo. "
            "Agrega imagenes etiquetadas de las clases faltantes antes de "
            "presentarlas como evaluacion completa."
        )


if __name__ == "__main__":
    main()
