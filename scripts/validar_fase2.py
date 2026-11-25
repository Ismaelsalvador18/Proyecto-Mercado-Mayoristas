#!/usr/bin/env python3
"""Valida la Fase 2: panel construido solo con los 24 departamentos.

Se ejecuta despues de 02_construir_panel.py y antes de cualquier análisis.
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

PROCESADOS = RAIZ / "data" / "processed"
PANEL = PROCESADOS / "panel_precios.csv"
COBERTURA = PROCESADOS / "cobertura_panel.csv"
UNIDADES = PROCESADOS / "unidades_panel.csv"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
MANIFIESTO_PANEL = MANIFIESTOS / "02_panel.json"
MANIFIESTO_VALIDACION = MANIFIESTOS / "02_validacion.json"

CLAVE = ["fecha", "departamento_id", "categoria"]


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def revisar_excluidos(datos: pd.DataFrame, columnas: list[str]) -> None:
    nombres = {nombre.casefold() for nombre, _ in EXCLUIDOS.values()}
    for columna in columnas:
        if columna not in datos.columns:
            continue
        encontrados = set(datos[columna].astype(str).str.casefold()) & nombres
        if encontrados:
            raise AssertionError(f"Aparecen mercados excluidos en {columna}: {sorted(encontrados)}")


def main() -> None:
    catalogo = cargar_catalogo()
    panel = pd.read_csv(PANEL, parse_dates=["fecha"], dtype={"departamento_id": str})
    cobertura = pd.read_csv(COBERTURA, dtype={"departamento_id": str})
    unidades = pd.read_csv(UNIDADES, dtype={"departamento_id": str})

    columnas_panel = ["fecha", "anio", "mes", "departamento_id", "departamento",
                      "region", "categoria", "precio_kg", "n_registros"]
    faltantes = [col for col in columnas_panel if col not in panel.columns]
    if faltantes:
        raise AssertionError(f"Faltan columnas en el panel: {faltantes}")
    if panel.isnull().any().any():
        raise AssertionError("El panel tiene nulos en alguna columna")
    if panel.duplicated(CLAVE).any():
        raise AssertionError("El panel tiene duplicados fecha-departamento-categoria")

    ids = set(panel["departamento_id"])
    esperados = set(catalogo["departamento_id"])
    if ids != esperados:
        raise AssertionError(
            f"Departamentos incorrectos. Faltan={sorted(esperados - ids)}; "
            f"sobran={sorted(ids - esperados)}"
        )

    revisar_excluidos(panel, ["departamento", "region"])
    revisar_excluidos(cobertura, ["departamento", "region"])
    revisar_excluidos(unidades, ["region"])

    if panel["categoria"].nunique() != 9:
        raise AssertionError("El panel no tiene 9 categorias")
    if panel.groupby(["departamento_id", "categoria"]).ngroups != 24 * 9:
        raise AssertionError("El panel no tiene las 216 series departamento-categoria")
    if len(cobertura) != 24 * 9:
        raise AssertionError("La cobertura no tiene 216 filas")
    if len(unidades) != 24 * 9:
        raise AssertionError("La tabla de unidades no tiene 216 filas")
    if int(cobertura["meses"].sum()) != len(panel):
        raise AssertionError("La cobertura no concilia con las filas del panel")

    # Cada ID debe tener un unico nombre y alias, y cada par debe ser unico.
    nombres = panel.groupby("departamento_id")[["departamento", "region"]].nunique()
    if (nombres > 1).any().any():
        raise AssertionError("Un departamento tiene mas de un nombre o alias")
    if unidades.duplicated(["departamento_id", "producto"]).any():
        raise AssertionError("La tabla de unidades tiene duplicados departamento-producto")
    if set(unidades["departamento_id"]) != esperados:
        raise AssertionError("La tabla de unidades no cubre los 24 departamentos")

    if not MANIFIESTO_PANEL.exists():
        raise AssertionError(f"No existe el manifiesto de panel: {MANIFIESTO_PANEL}")
    manifiesto = json.loads(MANIFIESTO_PANEL.read_text(encoding="utf-8"))
    if int(manifiesto["filas"]) != len(panel):
        raise AssertionError("El manifiesto no coincide con las filas del panel")
    if int(manifiesto["departamentos"]) != 24:
        raise AssertionError("El manifiesto no declara 24 departamentos")
    if int(manifiesto["series_observadas"]) != 216:
        raise AssertionError("El manifiesto no declara 216 series observadas")
    if manifiesto["salidas"]["panel_precios.csv"] != sha256_archivo(PANEL):
        raise AssertionError("El hash del panel no coincide con el manifiesto")
    if manifiesto["salidas"]["cobertura_panel.csv"] != sha256_archivo(COBERTURA):
        raise AssertionError("El hash de cobertura no coincide con el manifiesto")
    if manifiesto["salidas"]["unidades_panel.csv"] != sha256_archivo(UNIDADES):
        raise AssertionError("El hash de unidades no coincide con el manifiesto")

    salida = {
        "fase": "02_validacion",
        "ok": True,
        "filas": int(len(panel)),
        "departamentos": int(len(ids)),
        "categorias": int(panel["categoria"].nunique()),
        "series_potenciales": 24 * 9,
        "series_observadas": int(panel.groupby(["departamento_id", "categoria"]).ngroups),
        "fecha_min": panel["fecha"].min().strftime("%Y-%m-%d"),
        "fecha_max": panel["fecha"].max().strftime("%Y-%m-%d"),
        "filas_por_departamento": (
            panel.groupby("departamento_id").size().sort_index().astype(int).to_dict()
        ),
        "hashes": {
            "panel_precios.csv": sha256_archivo(PANEL),
            "cobertura_panel.csv": sha256_archivo(COBERTURA),
            "unidades_panel.csv": sha256_archivo(UNIDADES),
        },
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    MANIFIESTO_VALIDACION.write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 2 valida.")
    print(f"  Filas: {len(panel):,}")
    print(f"  Departamentos: {len(ids)}")
    print(f"  Series: {salida['series_observadas']} de {salida['series_potenciales']}")
    print(f"  Periodo: {salida['fecha_min']} a {salida['fecha_max']}")
    print(f"  Manifiesto: {MANIFIESTO_VALIDACION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
