#!/usr/bin/env python3
"""Cierre de la Fase 8: verifica que la corrida completa sea coherente."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalogo import EXCLUIDOS, cargar_catalogo  # noqa: E402

PROCESADOS = RAIZ / "data" / "processed"
ANALISIS = RAIZ / "outputs" / "analisis"
INFORME = RAIZ / "outputs" / "informe"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
CIERRE = MANIFIESTOS / "08_cierre.json"
VALIDACION = MANIFIESTOS / "08_validacion.json"

PATRONES_ANTIGUOS = [
    r"\b27 regiones\b",
    r"\b24 regiones\b",
    r"(?<![\d.,])243(?![\d.,])",
    r"(?<![\d.,])599(?![\d.,])",
    r"\b49[ .,]?041\b",
    r"\b49[ .,]?113\b",
    r"\b115[ .,]?294\b",
    r"\b85,5\b",
    r"\b65,8\b",
    r"\b0,017\b",
    r"\b0,247\b",
    r"\b53 %\b",
    r"papa, arroz, cebolla y huevos",
]


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

    for fase in [1, 2, 3, 4, 6, 7]:
        ruta = MANIFIESTOS / f"0{fase}_validacion.json"
        if not ruta.exists():
            raise AssertionError(f"Falta el manifiesto de validacion de Fase {fase}")
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if not datos.get("ok", False):
            raise AssertionError(f"Fase {fase} no esta aprobada")

    panel = pd.read_csv(PROCESADOS / "panel_precios.csv", dtype={"departamento_id": str})
    limpio = pd.read_csv(PROCESADOS / "panel_precios_limpio.csv", dtype={"departamento_id": str})
    atipicos = pd.read_csv(PROCESADOS / "atipicos_panel.csv", dtype={"departamento_id": str})
    if set(panel["departamento_id"]) != ids or set(limpio["departamento_id"]) != ids:
        raise AssertionError("El panel crudo o limpio no tiene los 24 departamentos")
    if panel.groupby(["departamento_id", "categoria"]).ngroups != 216:
        raise AssertionError("El panel crudo no tiene 216 series")
    if limpio.groupby(["departamento_id", "categoria"]).ngroups != 216:
        raise AssertionError("El panel limpio no tiene 216 series")
    if len(panel) != 43921 or len(limpio) != 43855 or len(atipicos) != 66:
        raise AssertionError("Los conteos finales de panel/limpieza no son los esperados")

    for ruta in sorted(ANALISIS.rglob("*.csv")) + sorted((INFORME / "tablas").glob("*.csv")):
        revisar_excluidos(pd.read_csv(ruta), str(ruta.relative_to(RAIZ)))

    ranking = pd.read_csv(ANALISIS / "06_modelos" / "torneo_nowcast_ranking.csv")
    if ranking.iloc[0]["modelo"] != "Random Forest":
        raise AssertionError("El ganador de nowcast no es Random Forest")
    predicciones = pd.read_csv(ANALISIS / "06_modelos" / "predicciones_holdout.csv")
    if len(predicciones) != 1590:
        raise AssertionError("El holdout final no tiene 1590 predicciones")

    tablas = sorted((INFORME / "tablas").glob("*.csv"))
    figuras = sorted((INFORME / "figuras").glob("*.png"))
    if len(tablas) != 16 or len(figuras) != 16:
        raise AssertionError("El informe no tiene 16 tablas y 16 figuras")

    docs = (RAIZ / "README.md").read_text(encoding="utf-8")
    for ruta in sorted((RAIZ / "docs").glob("*.md")):
        docs += "\n" + ruta.read_text(encoding="utf-8")
    for patron in PATRONES_ANTIGUOS:
        if re.search(patron, docs):
            raise AssertionError(f"Documentacion conserva un valor antiguo: {patron}")

    hashes = {}
    for ruta in [PROCESADOS / nombre for nombre in [
            "precios_ciudades_largo.csv", "panel_precios.csv",
            "panel_precios_limpio.csv", "atipicos_panel.csv"]]:
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)
    for ruta in tablas + figuras:
        hashes[str(ruta.relative_to(RAIZ))] = sha256_archivo(ruta)

    cierre = {
        "fase": "08_cierre",
        "ok": True,
        "departamentos": 24,
        "departamentos_sin_lima": 23,
        "filas_crudas": int(len(panel)),
        "filas_limpias": int(len(limpio)),
        "atipicos": int(len(atipicos)),
        "series": 216,
        "productos_holdout": 6,
        "modelos": 6,
        "predicciones_holdout": int(len(predicciones)),
        "tablas": len(tablas),
        "figuras": len(figuras),
        "hashes": hashes,
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    CIERRE.write_text(
        json.dumps(cierre, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    VALIDACION.write_text(
        json.dumps({k: v for k, v in cierre.items() if k != "hashes"},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 8 valida.")
    print("  Fases 1-4 y 6-7: aprobadas")
    print("  Panel: 24 departamentos | 216 series")
    print("  43 921 crudas -> 43 855 limpias | 66 atipicos")
    print("  Modelos: 6 productos | 6 modelos | 1 590 predicciones")
    print("  Informe: 16 tablas | 16 figuras")
    print(f"  Cierre: {CIERRE.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
