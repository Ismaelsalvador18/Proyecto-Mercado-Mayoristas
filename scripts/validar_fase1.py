#!/usr/bin/env python3
"""Valida la Fase 1: fuente larga restringida a los 24 departamentos.

Este script se ejecuta despues de 01_descargar_sisap.py. Falla con codigo
distinto de cero si la tabla larga no coincide exactamente con el catalogo.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogo import EXCLUIDOS, cargar_catalogo  # noqa: E402

LARGO = RAIZ / "data" / "processed" / "precios_ciudades_largo.csv"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
MANIFIESTO_FUENTE = MANIFIESTOS / "01_fuente.json"
MANIFIESTO_VALIDACION = MANIFIESTOS / "01_validacion.json"

COLUMNAS_CLAVE = [
    "departamento_id",
    "departamento",
    "region_codigo",
    "region",
    "codigo_producto",
    "producto",
    "anio",
    "mes",
    "valor",
]


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def main() -> None:
    catalogo = cargar_catalogo()
    datos = pd.read_csv(LARGO, dtype={"departamento_id": str, "region_codigo": str})
    datos["region_codigo"] = datos["region_codigo"].str.zfill(6)

    faltantes = [col for col in COLUMNAS_CLAVE if col not in datos.columns]
    if faltantes:
        raise AssertionError(f"Faltan columnas en la tabla larga: {faltantes}")
    nulos = datos[COLUMNAS_CLAVE].isnull().sum()
    if nulos.any():
        raise AssertionError(f"Hay nulos en columnas clave: {nulos[nulos > 0].to_dict()}")

    ids = set(datos["departamento_id"].astype(str))
    esperados = set(catalogo["departamento_id"])
    if ids != esperados:
        raise AssertionError(
            f"Departamentos incorrectos. Faltan={sorted(esperados - ids)}; "
            f"sobran={sorted(ids - esperados)}"
        )

    codigos = set(datos["region_codigo"])
    codigos_esperados = set(catalogo["codigo_sisap"])
    if codigos != codigos_esperados:
        raise AssertionError(
            f"Codigos SISAP incorrectos. Faltan={sorted(codigos_esperados - codigos)}; "
            f"sobran={sorted(codigos - codigos_esperados)}"
        )

    mapa = catalogo.set_index("codigo_sisap")
    for codigo, grupo in datos.groupby("region_codigo"):
        fila = mapa.loc[codigo]
        if set(grupo["departamento_id"]) != {fila.departamento_id}:
            raise AssertionError(f"ID inconsistente para el codigo {codigo}")
        if set(grupo["departamento"]) != {fila.departamento}:
            raise AssertionError(f"Nombre oficial inconsistente para el codigo {codigo}")
        if set(grupo["region"]) != {fila.departamento_ascii}:
            raise AssertionError(f"Alias ASCII inconsistente para el codigo {codigo}")

    nombres_excluidos = {nombre.casefold() for nombre, _ in EXCLUIDOS.values()}
    for columna in ["departamento", "region"]:
        encontrados = set(datos[columna].astype(str).str.casefold()) & nombres_excluidos
        if encontrados:
            raise AssertionError(f"Aparecen mercados excluidos en {columna}: {sorted(encontrados)}")
    if set(datos["region_codigo"]) & set(EXCLUIDOS):
        raise AssertionError("Aparecen codigos SISAP de mercados excluidos")

    categorias = datos["codigo_producto"].astype(str).str.zfill(6).str[:4].nunique()
    variedades = datos["producto"].nunique()
    if categorias != 9:
        raise AssertionError(f"Se esperaban 9 categorias; hay {categorias}")
    if datos["departamento_id"].nunique() != 24:
        raise AssertionError("La tabla larga no tiene 24 departamentos")

    if not MANIFIESTO_FUENTE.exists():
        raise AssertionError(f"No existe el manifiesto de fuente: {MANIFIESTO_FUENTE}")
    manifiesto = json.loads(MANIFIESTO_FUENTE.read_text(encoding="utf-8"))
    if int(manifiesto["filas"]) != len(datos):
        raise AssertionError(
            f"El manifiesto declara {manifiesto['filas']} filas; la tabla tiene {len(datos)}"
        )
    hash_real = sha256_archivo(LARGO)
    if manifiesto["salida_sha256"] != hash_real:
        raise AssertionError("El hash de la tabla larga no coincide con el manifiesto")

    por_departamento = (
        datos.groupby("departamento_id").size().sort_index().astype(int).to_dict()
    )
    salida = {
        "fase": "01_validacion",
        "ok": True,
        "filas": int(len(datos)),
        "departamentos": int(len(ids)),
        "categorias": int(categorias),
        "variedades": int(variedades),
        "anio_min": int(datos["anio"].min()),
        "anio_max": int(datos["anio"].max()),
        "filas_por_departamento": por_departamento,
        "departamentos_ids": sorted(ids),
        "salida_sha256": hash_real,
    }
    MANIFIESTO_VALIDACION.write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 1 valida.")
    print(f"  Filas: {len(datos):,}")
    print(f"  Departamentos: {len(ids)}")
    print(f"  Categorias: {categorias} | Variedades: {variedades}")
    print(f"  Periodo: {datos.anio.min()}-{datos.anio.max()}")
    print(f"  Manifiesto: {MANIFIESTO_VALIDACION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
