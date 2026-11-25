#!/usr/bin/env python3
"""Limpia valores incompatibles con el entorno del precio.

El SISAP tiene errores de transcripcion tipicos: un precio de 145,34 S/kg de
cebolla cuando la mediana nacional ese mes es 2,36 (desplazamiento de punto
decimal). No se imputan: se marcan como faltantes y se reportan.

Regla (doble, para no confundir un pico real con un error):
  1. Comparacion local: precio / mediana movil de +-3 meses. Se descarta si
     el cociente es > 4 o < 0,25.
  2. Comparacion nacional: precio / mediana del mismo producto y mes entre
     departamentos. Se descarta si el cociente es > 5 o < 0,2.

Entrada : data/processed/panel_precios.csv
Salidas : data/processed/panel_precios_limpio.csv
          data/processed/atipicos_panel.csv
          outputs/_manifests/03_limpieza.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalogo import cargar_catalogo  # noqa: E402

ENTRADA = RAIZ / "data" / "processed" / "panel_precios.csv"
SALIDAS = RAIZ / "data" / "processed"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"

CLAVE = ["fecha", "departamento_id", "categoria"]


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def main() -> None:
    catalogo = cargar_catalogo()
    panel = pd.read_csv(
        ENTRADA,
        parse_dates=["fecha"],
        dtype={"departamento_id": str},
    )
    requeridas = ["fecha", "departamento_id", "departamento", "region",
                  "categoria", "precio_kg"]
    faltantes = [col for col in requeridas if col not in panel.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el panel: {faltantes}")

    ids = set(panel["departamento_id"])
    if ids != set(catalogo["departamento_id"]):
        raise ValueError("El panel de entrada no tiene exactamente los 24 departamentos")
    panel = panel.sort_values(
        ["departamento_id", "categoria", "fecha"]
    ).reset_index(drop=True)

    # Las medianas se calculan sobre los 24 departamentos, ya filtrados en 02.
    panel["mediana_nacional"] = (
        panel.groupby(["categoria", "fecha"])["precio_kg"].transform("median")
    )
    panel["ratio_nacional"] = panel["precio_kg"] / panel["mediana_nacional"]

    grupo = panel.groupby(["departamento_id", "categoria"])["precio_kg"]
    panel["mediana_local"] = grupo.transform(
        lambda s: s.rolling(7, center=True, min_periods=3).median()
    )
    panel["ratio_local"] = panel["precio_kg"] / panel["mediana_local"]

    atipico = (
        (panel["ratio_local"] > 4)
        | (panel["ratio_local"] < 0.25)
        | (panel["ratio_nacional"] > 5)
        | (panel["ratio_nacional"] < 0.2)
        | (panel["precio_kg"] < 0.05)
    )

    panel["atipico"] = atipico
    panel["precio_limpio"] = panel["precio_kg"].where(~atipico)

    detalle = panel.loc[atipico, [
        "fecha", "departamento_id", "departamento", "region", "categoria",
        "precio_kg", "mediana_local", "mediana_nacional", "ratio_local",
        "ratio_nacional",
    ]].copy()
    detalle.to_csv(SALIDAS / "atipicos_panel.csv", index=False, encoding="utf-8")

    limpio = panel.drop(columns=[
        "precio_kg", "mediana_local", "mediana_nacional", "ratio_local",
        "ratio_nacional", "atipico",
    ])
    limpio = limpio.rename(columns={"precio_limpio": "precio_kg"})
    limpio = limpio.dropna(subset=["precio_kg"])

    # El panel limpio conserva la unidad departamental y las 216 series.
    ids_limpios = set(limpio["departamento_id"])
    if ids_limpios != set(catalogo["departamento_id"]):
        raise ValueError(
            "La limpieza elimino departamentos: "
            f"faltan={sorted(set(catalogo['departamento_id']) - ids_limpios)}"
        )
    if limpio.duplicated(CLAVE).any():
        raise ValueError("El panel limpio tiene duplicados fecha-departamento-categoria")
    if limpio[["precio_kg", "departamento_id", "categoria", "fecha"]].isnull().any().any():
        raise ValueError("El panel limpio tiene nulos en columnas clave")
    if limpio.groupby(["departamento_id", "categoria"]).ngroups != 24 * 9:
        raise ValueError("La limpieza perdio series departamento-categoria")

    limpio.to_csv(SALIDAS / "panel_precios_limpio.csv", index=False, encoding="utf-8")

    salida_manifiesto = MANIFIESTOS / "03_limpieza.json"
    manifiesto = {
        "fase": "03_limpieza",
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "entrada": str(ENTRADA.relative_to(RAIZ)),
        "entrada_sha256": sha256_archivo(ENTRADA),
        "catalogo_sha256": sha256_archivo(
            RAIZ / "data" / "reference" / "departamentos.csv"
        ),
        "filas_antes": int(len(panel)),
        "filas_despues": int(len(limpio)),
        "atipicos": int(atipico.sum()),
        "atipicos_pct": float(100 * atipico.mean()),
        "departamentos": int(len(ids_limpios)),
        "series_observadas": int(limpio.groupby(["departamento_id", "categoria"]).ngroups),
        "filas_por_departamento": (
            limpio.groupby("departamento_id").size().sort_index().astype(int).to_dict()
        ),
        "atipicos_por_departamento": (
            detalle.groupby("departamento_id").size().sort_index().astype(int).to_dict()
        ),
        "salidas": {
            "panel_precios_limpio.csv": sha256_archivo(
                SALIDAS / "panel_precios_limpio.csv"
            ),
            "atipicos_panel.csv": sha256_archivo(SALIDAS / "atipicos_panel.csv"),
        },
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    salida_manifiesto.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Filas originales : {len(panel):,}")
    print(f"Atipicos marcados: {int(atipico.sum()):,} ({100 * atipico.mean():.2f} %)")
    print("\nPor producto:")
    print(detalle.groupby("categoria").size().sort_values(ascending=False).to_string())
    print("\nEjemplos mas extremos:")
    print(
        detalle.assign(ratio_nacional=detalle["ratio_nacional"].round(1))
        .nlargest(8, "ratio_nacional")
        [["fecha", "departamento", "categoria", "precio_kg",
          "mediana_nacional", "ratio_nacional"]]
        .to_string(index=False)
    )
    print(f"\nArchivo limpio: {len(limpio):,} filas -> "
          f"data/processed/panel_precios_limpio.csv")
    print(f"Manifiesto: {salida_manifiesto.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
