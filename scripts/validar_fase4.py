#!/usr/bin/env python3
"""Valida la Fase 4: EDA, estructura y estacionariedad con 24 departamentos."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogo import EXCLUIDOS, cargar_catalogo  # noqa: E402
from config import NUCLEO, PANEL_LIMPIO, PRODUCTOS  # noqa: E402

CALIDAD = RAIZ / "outputs" / "analisis" / "03_calidad"
ESTRUCTURA = RAIZ / "outputs" / "analisis" / "04_estructura"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
MANIFIESTO_LIMPIEZA = MANIFIESTOS / "03_limpieza.json"
MANIFIESTO_EDA = MANIFIESTOS / "04_eda.json"
MANIFIESTO_VALIDACION = MANIFIESTOS / "04_validacion.json"

ZONAS = {"Costa", "Sierra", "Selva"}


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(CALIDAD / nombre)


def leer_estructura(nombre: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(ESTRUCTURA / nombre, **kwargs)


def revisar_excluidos(datos: pd.DataFrame, etiqueta: str) -> None:
    nombres = {nombre.casefold() for nombre, _ in EXCLUIDOS.values()}
    for columna in datos.select_dtypes(include=["object", "string"]).columns:
        valores = set(datos[columna].dropna().astype(str).str.casefold())
        encontrados = valores & nombres
        if encontrados:
            raise AssertionError(f"{etiqueta}: mercados excluidos en {columna}: {sorted(encontrados)}")


def main() -> None:
    catalogo = cargar_catalogo()
    ids = set(catalogo["departamento_id"])
    permitidos = set(catalogo["departamento_ascii"])
    sin_lima = permitidos - {"Lima"}

    panel = pd.read_csv(PANEL_LIMPIO, parse_dates=["fecha"], dtype={"departamento_id": str})
    if set(panel["departamento_id"]) != ids:
        raise AssertionError("El panel limpio no tiene los 24 departamentos")
    if panel.groupby(["departamento_id", "categoria"]).ngroups != 216:
        raise AssertionError("El panel limpio no tiene 216 series")
    revisar_excluidos(panel, "panel limpio")

    # Archivos de calidad y estructura.
    archivos = sorted(CALIDAD.glob("*.csv")) + sorted(ESTRUCTURA.glob("*.csv"))
    for ruta in archivos:
        revisar_excluidos(pd.read_csv(ruta), ruta.name)

    variables = leer("tabla_variables.csv")
    if variables.loc[variables.variable == "region", "unidad"].iloc[0] != "24 departamentos":
        raise AssertionError("La tabla de variables no declara 24 departamentos")

    for nombre in ["descriptivos_producto.csv", "normalidad.csv", "transformacion_log.csv"]:
        datos = leer(nombre)
        if set(datos["producto"]) != set(PRODUCTOS):
            raise AssertionError(f"{nombre} no contiene las 9 categorias")
        if len(datos) != 9:
            raise AssertionError(f"{nombre} no tiene 9 filas")

    est_prod = leer("estacionariedad_producto.csv")
    if len(est_prod) != 12:
        raise AssertionError("Estacionariedad producto no tiene 12 filas")
    if set(est_prod["producto"]) != set(NUCLEO) or set(est_prod["serie"]) != {"Puno", "Lima"}:
        raise AssertionError("Estacionariedad producto no cubre Puno y Lima por producto")

    est_reg = leer("estacionariedad_region.csv")
    if not set(est_reg["region"]) <= sin_lima:
        raise AssertionError("Estacionariedad regional contiene Lima o una region no oficial")
    if est_reg.duplicated(["producto", "region"]).any():
        raise AssertionError("Estacionariedad regional tiene duplicados")
    if est_reg.groupby("producto")["region"].nunique().max() > 23:
        raise AssertionError("Estacionariedad regional supera los 23 departamentos sin Lima")
    if set(est_reg["producto"]) != set(NUCLEO):
        raise AssertionError("Estacionariedad regional no cubre los seis productos nucleo")

    est_res = leer("estacionariedad_resumen.csv")
    if set(est_res["producto"]) != set(NUCLEO):
        raise AssertionError("El resumen de estacionariedad no cubre los seis productos")
    if (est_res["departamentos"] > 23).any():
        raise AssertionError("El resumen de estacionariedad supera 23 departamentos")
    if ((est_res["i0"] + est_res["i1"] + est_res["mixto"]) != est_res["departamentos"]).any():
        raise AssertionError("El resumen de estacionariedad no suma sus categorias")

    seleccion = leer("seleccion_rezagos.csv")
    if set(seleccion["producto"]) != set(NUCLEO):
        raise AssertionError("La seleccion de rezagos no cubre los seis productos")
    for columna in ["q_aic", "q_bic", "q_elegido"]:
        if not seleccion[columna].between(0, 4).all():
            raise AssertionError(f"{columna} fuera del rango 0-4")

    corr_lima = leer_estructura("correlacion_lima.csv")
    if not set(corr_lima["region"]) <= sin_lima:
        raise AssertionError("Correlacion con Lima contiene Lima o una region no oficial")
    if corr_lima.duplicated(["producto", "region"]).any():
        raise AssertionError("Correlacion con Lima tiene duplicados")
    if (corr_lima["n"] < 60).any():
        raise AssertionError("Correlacion con Lima incluye pares con n < 60")
    if corr_lima.groupby("producto")["region"].nunique().max() > 23:
        raise AssertionError("Correlacion con Lima supera los 23 departamentos sin Lima")

    papa = leer_estructura("correlacion_papa.csv", index_col=0)
    if set(papa.index) != set(papa.columns):
        raise AssertionError("La matriz de papa no es cuadrada")
    if not set(papa.index) <= permitidos:
        raise AssertionError("La matriz de papa contiene regiones no oficiales")
    if papa.shape != (10, 10):
        raise AssertionError("La matriz de papa no tiene el tamano esperado")

    puno = leer_estructura("correlacion_puno.csv", index_col=0)
    if set(puno.index) != set(PRODUCTOS) or set(puno.columns) != set(PRODUCTOS):
        raise AssertionError("La matriz de Puno no cubre las 9 categorias")

    geo = leer_estructura("geografia_integracion.csv")
    if len(geo) != 23 or set(geo["region"]) != sin_lima:
        raise AssertionError("La geografia no cubre los 23 departamentos sin Lima")
    if geo["region"].duplicated().any():
        raise AssertionError("La geografia tiene departamentos duplicados")
    if not set(geo["zona"]) <= ZONAS:
        raise AssertionError("La geografia contiene zonas invalidas")
    if not geo["r_lima"].between(-1, 1).all():
        raise AssertionError("La geografia contiene correlaciones fuera de rango")
    if not (geo["distancia_km"] > 0).all():
        raise AssertionError("La geografia contiene distancias invalidas")

    geo_modelos = leer_estructura("geografia_modelos.csv")
    if len(geo_modelos) != 4:
        raise AssertionError("Los modelos geograficos no tienen 4 filas")

    if MANIFIESTO_LIMPIEZA.exists():
        limpieza = json.loads(MANIFIESTO_LIMPIEZA.read_text(encoding="utf-8"))
        verificacion = leer("verificacion_limpieza.csv").set_index("metrica")["valor"]
        if float(verificacion.loc["filas antes"]) != limpieza["filas_antes"]:
            raise AssertionError("La verificacion no coincide con la limpieza")
        if float(verificacion.loc["filas despues"]) != limpieza["filas_despues"]:
            raise AssertionError("La verificacion no coincide con la limpieza")
        if float(verificacion.loc["atipicos marcados"]) != limpieza["atipicos"]:
            raise AssertionError("La verificacion no coincide con los atipicos")

    hashes = {str(ruta.relative_to(RAIZ)): sha256_archivo(ruta) for ruta in archivos}
    manifiesto = {
        "fase": "04_eda",
        "catalogo_sha256": sha256_archivo(
            RAIZ / "data" / "reference" / "departamentos.csv"
        ),
        "panel_limpio_sha256": sha256_archivo(PANEL_LIMPIO),
        "departamentos": int(len(ids)),
        "departamentos_con_geografia": int(len(geo)),
        "productos": int(len(PRODUCTOS)),
        "estacionariedad_filas": int(len(est_reg)),
        "correlacion_lima_filas": int(len(corr_lima)),
        "archivos": hashes,
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    MANIFIESTO_EDA.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validacion = {
        "fase": "04_validacion",
        "ok": True,
        "departamentos": 24,
        "departamentos_sin_lima": 23,
        "productos": 9,
        "estacionariedad_filas": int(len(est_reg)),
        "correlacion_lima_filas": int(len(corr_lima)),
        "archivos_validados": len(archivos),
        "hashes": hashes,
    }
    MANIFIESTO_VALIDACION.write_text(
        json.dumps(validacion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 4 valida.")
    print(f"  Panel: 24 departamentos | 216 series")
    print(f"  Estacionariedad: {len(est_reg)} filas regionales")
    print(f"  Correlacion con Lima: {len(corr_lima)} pares")
    print(f"  Geografia: {len(geo)} departamentos sin Lima")
    print(f"  Archivos CSV validados: {len(archivos)}")
    print(f"  Manifiesto: {MANIFIESTO_VALIDACION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
