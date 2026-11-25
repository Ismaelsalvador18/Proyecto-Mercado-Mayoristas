#!/usr/bin/env python3
"""Construye el panel mensual de precios mayoristas por departamento y producto.

Entrada : data/processed/precios_ciudades_largo.csv
Salidas : data/processed/panel_precios.csv       (formato largo tidy)
          data/processed/cobertura_panel.csv     (cobertura por producto-departamento)
          data/processed/unidades_panel.csv      (unidad dominante elegida)
          outputs/_manifests/02_panel.json       (contrato de la fase)

Reglas:
  1. Se une la tabla larga con el catalogo por ``departamento_id``. No se
     aceptan nombres fuera del catalogo.
  2. Se elige una variedad comercial por producto (la de mejor cobertura
     conjunta entre departamentos y anos).
  3. Para cada departamento y producto se toma la unidad de medida dominante
     en el tiempo (kilogramo, arroba, saco...). Asi no se mezclan
     presentaciones.
  4. El precio se normaliza a soles por kilogramo dividiendo por la
     equivalencia en kg que publica el SISAP.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalogo import cargar_catalogo  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent.parent
ENTRADA = RAIZ / "data" / "processed" / "precios_ciudades_largo.csv"
SALIDAS = RAIZ / "data" / "processed"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"

VARIEDADES = {
    "Papa canchan": "Papa",
    "Cebolla cabeza roja": "Cebolla",
    "Tomate marzano": "Tomate",
    "Zanahoria": "Zanahoria",
    "Limon sutil bolsa": "Limon",
    "Arroz superior": "Arroz",
    "Huevos rosados": "Huevos",
    "Azucar rubia nacional (granel)": "Azucar",
    "Aceite clásico botella x 1l": "Aceite",
}

CLAVE_PANEL = ["fecha", "departamento_id", "categoria"]


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def unidad_dominante(datos: pd.DataFrame) -> pd.DataFrame:
    conteo = (
        datos.groupby(
            ["departamento_id", "region", "producto", "unidad", "equivalencia_kg"]
        )
        .size()
        .reset_index(name="n")
    )
    # En empate, prefiere kilogramo (equivalencia 1).
    conteo["prioridad"] = np.where(
        (conteo["unidad"] == "Kilogramo") & (conteo["equivalencia_kg"] == 1), 0, 1
    )
    conteo = conteo.sort_values(
        ["departamento_id", "producto", "n", "prioridad"],
        ascending=[True, True, False, True],
    )
    return conteo.drop_duplicates(["departamento_id", "producto"])


def main() -> None:
    catalogo = cargar_catalogo()
    crudo = pd.read_csv(ENTRADA, dtype={"departamento_id": str, "region_codigo": str})
    crudo["region_codigo"] = crudo["region_codigo"].str.zfill(6)

    requeridas = ["departamento_id", "codigo_producto", "producto", "unidad",
                  "equivalencia_kg", "anio", "mes", "valor", "region_codigo"]
    faltantes = [col for col in requeridas if col not in crudo.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en la tabla larga: {faltantes}")

    datos = crudo[crudo["producto"].isin(VARIEDADES)].copy()
    datos = datos.drop(columns=["departamento", "region"], errors="ignore")

    # El catalogo es la unica fuente de nombres y codigos canonicos.
    mapa = catalogo[
        ["departamento_id", "departamento", "departamento_ascii", "codigo_sisap"]
    ].rename(columns={"departamento_ascii": "region"})
    datos = datos.merge(
        mapa,
        on="departamento_id",
        how="inner",
        validate="many_to_one",
    )
    if set(datos["departamento_id"]) != set(catalogo["departamento_id"]):
        raise ValueError("La tabla larga no cubre exactamente los 24 departamentos")
    if not (datos["region_codigo"] == datos["codigo_sisap"]).all():
        raise ValueError("El codigo SISAP no coincide con el catalogo")

    datos["categoria"] = datos["producto"].map(VARIEDADES)

    elegidas = unidad_dominante(datos)
    datos = datos.merge(
        elegidas[["departamento_id", "producto", "unidad", "equivalencia_kg"]],
        on=["departamento_id", "producto", "unidad", "equivalencia_kg"],
        how="inner",
        validate="many_to_one",
    )

    datos["precio_kg"] = datos["valor"] / datos["equivalencia_kg"].replace(0, np.nan)
    datos["fecha"] = pd.to_datetime(
        dict(year=datos["anio"], month=datos["mes"], day=1)
    )

    # Duplicados residuales: promedio mensual.
    panel = (
        datos.groupby(
            ["fecha", "anio", "mes", "departamento_id", "departamento", "region",
             "categoria"],
            as_index=False,
        )
        .agg(precio_kg=("precio_kg", "mean"), n_registros=("precio_kg", "size"))
    )
    panel = panel.sort_values(
        ["categoria", "departamento_id", "fecha"]
    ).reset_index(drop=True)

    ids = set(panel["departamento_id"])
    if ids != set(catalogo["departamento_id"]):
        raise ValueError(
            "El panel no conserva los 24 departamentos: "
            f"faltan={sorted(set(catalogo['departamento_id']) - ids)}, "
            f"sobran={sorted(ids - set(catalogo['departamento_id']))}"
        )
    if panel.duplicated(CLAVE_PANEL).any():
        raise ValueError("El panel tiene duplicados fecha-departamento-categoria")
    if panel[["precio_kg", "departamento_id", "categoria", "fecha"]].isnull().any().any():
        raise ValueError("El panel tiene nulos en columnas clave")

    panel.to_csv(SALIDAS / "panel_precios.csv", index=False, encoding="utf-8")
    elegidas.to_csv(SALIDAS / "unidades_panel.csv", index=False, encoding="utf-8")

    cobertura = (
        panel.groupby(["categoria", "departamento_id", "departamento", "region"])
        .agg(
            meses=("precio_kg", "size"),
            primer_mes=("fecha", "min"),
            ultimo_mes=("fecha", "max"),
            precio_medio=("precio_kg", "mean"),
        )
        .reset_index()
    )
    cobertura["anios_cubiertos"] = cobertura["meses"] / 12
    cobertura.to_csv(SALIDAS / "cobertura_panel.csv", index=False, encoding="utf-8")

    series_potenciales = len(catalogo) * panel["categoria"].nunique()
    series_observadas = panel.groupby(["departamento_id", "categoria"]).ngroups
    manifiesto = {
        "fase": "02_panel",
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "entrada": str(ENTRADA.relative_to(RAIZ)),
        "entrada_sha256": sha256_archivo(ENTRADA),
        "catalogo_sha256": sha256_archivo(
            RAIZ / "data" / "reference" / "departamentos.csv"
        ),
        "filas": int(len(panel)),
        "departamentos": int(len(ids)),
        "categorias": int(panel["categoria"].nunique()),
        "series_potenciales": int(series_potenciales),
        "series_observadas": int(series_observadas),
        "fecha_min": panel["fecha"].min().strftime("%Y-%m-%d"),
        "fecha_max": panel["fecha"].max().strftime("%Y-%m-%d"),
        "filas_por_departamento": (
            panel.groupby("departamento_id").size().sort_index().astype(int).to_dict()
        ),
        "salidas": {
            "panel_precios.csv": sha256_archivo(SALIDAS / "panel_precios.csv"),
            "cobertura_panel.csv": sha256_archivo(SALIDAS / "cobertura_panel.csv"),
            "unidades_panel.csv": sha256_archivo(SALIDAS / "unidades_panel.csv"),
        },
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    destino = MANIFIESTOS / "02_panel.json"
    destino.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Panel: {panel.shape[0]:,} filas | {len(ids)} departamentos "
          f"| {panel.categoria.nunique()} productos")
    print(f"Series: {series_observadas} observadas de {series_potenciales} potenciales")
    print(f"Ventana: {panel.fecha.min():%Y-%m} a {panel.fecha.max():%Y-%m}")
    print("\nMeses por producto (resumen entre departamentos):")
    resumen = (
        cobertura.groupby("categoria")["meses"]
        .agg(["min", "median", "max"])
        .sort_values("median", ascending=False)
    )
    print(resumen.round(0).to_string())
    print("\nCobertura de Puno:")
    print(
        cobertura[cobertura.region == "Puno"]
        .sort_values("meses", ascending=False)
        .to_string(index=False)
    )
    print(f"\nManifiesto: {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
