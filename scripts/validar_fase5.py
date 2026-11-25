#!/usr/bin/env python3
"""Valida la Fase 5: ECM, diagnostico, asimetria y robustez con 24 departamentos."""

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
from config import MIN_MESES, NUCLEO, PANEL_LIMPIO  # noqa: E402

ECM = RAIZ / "outputs" / "analisis" / "05_ecm"
ROBUSTEZ = RAIZ / "outputs" / "analisis" / "07_robustez"
CALIDAD = RAIZ / "outputs" / "analisis" / "03_calidad"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
MANIFIESTO_ECM = MANIFIESTOS / "05_ecm.json"
MANIFIESTO_VALIDACION = MANIFIESTOS / "05_validacion.json"


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


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

    archivos = sorted(ECM.glob("*.csv")) + sorted(ROBUSTEZ.glob("*.csv"))
    for ruta in archivos:
        revisar_excluidos(pd.read_csv(ruta), ruta.name)

    puno = pd.read_csv(ECM / "ecm_puno.csv")
    if set(puno["producto"]) != set(NUCLEO):
        raise AssertionError("El ECM de Puno no cubre los seis productos")
    if (puno["n"] < MIN_MESES).any():
        raise AssertionError("El ECM de Puno tiene muestras menores al minimo")
    if not puno["gamma"].lt(0).all():
        raise AssertionError("El ECM de Puno tiene gamma no negativo")
    if not puno["vida_media"].notna().all():
        raise AssertionError("El ECM de Puno tiene vidas medias faltantes")

    regiones = pd.read_csv(ECM / "ecm_region.csv")
    if not set(regiones["region"]) <= sin_lima:
        raise AssertionError("El ECM regional contiene Lima o una region no oficial")
    if regiones.duplicated(["producto", "region"]).any():
        raise AssertionError("El ECM regional tiene duplicados producto-region")
    if regiones.groupby("producto")["region"].nunique().max() > 23:
        raise AssertionError("El ECM regional supera los 23 departamentos sin Lima")
    if set(regiones["producto"]) != set(NUCLEO):
        raise AssertionError("El ECM regional no cubre los seis productos")
    if (regiones["n"] < MIN_MESES).any():
        raise AssertionError("El ECM regional tiene muestras menores al minimo")

    producto = pd.read_csv(ECM / "ecm_producto.csv")
    if set(producto["producto"]) != set(NUCLEO):
        raise AssertionError("El resumen ECM no cubre los seis productos")
    if (producto["pares"] > 23).any():
        raise AssertionError("El resumen ECM declara mas de 23 pares")
    if (producto["pares_significativos"] > producto["pares"]).any():
        raise AssertionError("Hay mas pares significativos que pares estimados")
    conteo = regiones.groupby("producto")["region"].nunique()
    for fila in producto.itertuples(index=False):
        if int(conteo.loc[fila.producto]) != int(fila.pares):
            raise AssertionError(f"Pares inconsistentes para {fila.producto}")

    seleccion = pd.read_csv(CALIDAD / "seleccion_rezagos.csv")
    qs = dict(zip(seleccion["producto"], seleccion["q_elegido"]))
    for fila in puno.itertuples(index=False):
        if int(qs[fila.producto]) != int(fila.q):
            raise AssertionError(f"q inconsistente para {fila.producto}")

    asimetria = pd.read_csv(ECM / "asimetria.csv")
    if set(asimetria["producto"]) != set(NUCLEO):
        raise AssertionError("La asimetria no cubre los seis productos")
    if not asimetria["p_igualdad"].between(0, 1).all():
        raise AssertionError("La asimetria tiene p-valores invalidos")

    diagnostico = pd.read_csv(ECM / "diagnostico_ecm.csv")
    if set(diagnostico["producto"]) != set(NUCLEO):
        raise AssertionError("El diagnostico no cubre los seis productos")
    n_puno = dict(zip(puno["producto"], puno["n"]))
    for fila in diagnostico.itertuples(index=False):
        if int(n_puno[fila.producto]) != int(fila.n):
            raise AssertionError(f"n inconsistente en diagnostico para {fila.producto}")
    for columna in ["ljung_box_4", "ljung_box_8", "breusch_pagan", "jarque_bera"]:
        if not diagnostico[columna].between(0, 1).all():
            raise AssertionError(f"p-valor invalido en {columna}")

    robustez = pd.read_csv(ROBUSTEZ / "ecm_robustez.csv")
    escenarios = set(robustez["escenario"])
    esperados = {"base", "q0", "q1", "q2", "sin_shock", "ventana_2008"}
    if not esperados <= escenarios:
        raise AssertionError(f"Faltan escenarios de robustez: {sorted(esperados - escenarios)}")
    base = robustez[robustez["escenario"] == "base"]
    if set(base["producto"]) != set(NUCLEO):
        raise AssertionError("La robustez base no cubre los seis productos")
    if base["vida_media"].isna().any():
        raise AssertionError("La robustez base tiene vidas medias faltantes")
    if not robustez["q"].between(0, 2).all():
        raise AssertionError("La robustez contiene q fuera de 0-2")

    hashes = {str(ruta.relative_to(RAIZ)): sha256_archivo(ruta) for ruta in archivos}
    manifiesto = {
        "fase": "05_ecm",
        "catalogo_sha256": sha256_archivo(
            RAIZ / "data" / "reference" / "departamentos.csv"
        ),
        "panel_limpio_sha256": sha256_archivo(PANEL_LIMPIO),
        "departamentos": 24,
        "departamentos_sin_lima": 23,
        "productos": 6,
        "ecm_puno_filas": int(len(puno)),
        "ecm_region_filas": int(len(regiones)),
        "robustez_filas": int(len(robustez)),
        "archivos": hashes,
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    MANIFIESTO_ECM.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    validacion = {
        "fase": "05_validacion",
        "ok": True,
        "departamentos": 24,
        "departamentos_sin_lima": 23,
        "productos": 6,
        "ecm_puno_filas": int(len(puno)),
        "ecm_region_filas": int(len(regiones)),
        "pares_max": int(regiones.groupby("producto")["region"].nunique().max()),
        "robustez_filas": int(len(robustez)),
        "archivos_validados": len(archivos),
        "hashes": hashes,
    }
    MANIFIESTO_VALIDACION.write_text(
        json.dumps(validacion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 5 valida.")
    print(f"  ECM Puno: {len(puno)} productos")
    print(f"  ECM regional: {len(regiones)} pares | maximo {validacion['pares_max']} por producto")
    print(f"  Robustez: {len(robustez)} filas | escenarios {len(escenarios)}")
    print(f"  Archivos CSV validados: {len(archivos)}")
    print(f"  Manifiesto: {MANIFIESTO_VALIDACION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
