#!/usr/bin/env python3
"""Valida la Fase 3: limpieza sobre el panel de 24 departamentos.

Comprueba que la limpieza sea una particion exacta del panel crudo:
cada observacion termina en el panel limpio o en la tabla de atipicos,
sin perder filas ni duplicar claves.
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
CRUDO = PROCESADOS / "panel_precios.csv"
LIMPIO = PROCESADOS / "panel_precios_limpio.csv"
ATIPICOS = PROCESADOS / "atipicos_panel.csv"
MANIFIESTOS = RAIZ / "outputs" / "_manifests"
MANIFIESTO_LIMPIEZA = MANIFIESTOS / "03_limpieza.json"
MANIFIESTO_VALIDACION = MANIFIESTOS / "03_validacion.json"

CLAVE = ["fecha", "departamento_id", "categoria"]


def sha256_archivo(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def claves(datos: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, datos[CLAVE].itertuples(index=False, name=None)))


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
    crudo = pd.read_csv(CRUDO, parse_dates=["fecha"], dtype={"departamento_id": str})
    limpio = pd.read_csv(LIMPIO, parse_dates=["fecha"], dtype={"departamento_id": str})
    atipicos = pd.read_csv(ATIPICOS, parse_dates=["fecha"], dtype={"departamento_id": str})
    ids = set(catalogo["departamento_id"])

    for nombre, datos in [("crudo", crudo), ("limpio", limpio), ("atipicos", atipicos)]:
        faltantes = [col for col in CLAVE + ["departamento", "region"] if col not in datos.columns]
        if faltantes:
            raise AssertionError(f"Faltan columnas en {nombre}: {faltantes}")
        observados = set(datos["departamento_id"])
        if not observados <= ids:
            raise AssertionError(
                f"{nombre} contiene departamentos fuera del catalogo: "
                f"{sorted(observados - ids)}"
            )
        revisar_excluidos(datos, ["departamento", "region"])

    if set(crudo["departamento_id"]) != ids:
        raise AssertionError("El panel crudo no tiene los 24 departamentos")
    if set(limpio["departamento_id"]) != ids:
        raise AssertionError("El panel limpio no tiene los 24 departamentos")
    if len(limpio) != len(crudo) - len(atipicos):
        raise AssertionError(
            f"La particion no cuadra: crudo={len(crudo)}, limpio={len(limpio)}, "
            f"atipicos={len(atipicos)}"
        )
    if crudo.duplicated(CLAVE).any():
        raise AssertionError("El panel crudo tiene duplicados")
    if limpio.duplicated(CLAVE).any():
        raise AssertionError("El panel limpio tiene duplicados")
    if atipicos.duplicated(CLAVE).any():
        raise AssertionError("La tabla de atipicos tiene duplicados")
    if limpio.isnull().any().any():
        raise AssertionError("El panel limpio tiene nulos")
    if limpio["categoria"].nunique() != 9:
        raise AssertionError("El panel limpio no tiene 9 categorias")
    if limpio.groupby(["departamento_id", "categoria"]).ngroups != 24 * 9:
        raise AssertionError("El panel limpio no conserva las 216 series")

    k_crudo = claves(crudo)
    k_limpio = claves(limpio)
    k_atipicos = claves(atipicos)
    if k_limpio & k_atipicos:
        raise AssertionError("Hay claves presentes a la vez en limpio y atipicos")
    if (k_limpio | k_atipicos) != k_crudo:
        raise AssertionError("La limpieza no cubre exactamente todas las claves del crudo")

    if not MANIFIESTO_LIMPIEZA.exists():
        raise AssertionError(f"No existe el manifiesto de limpieza: {MANIFIESTO_LIMPIEZA}")
    manifiesto = json.loads(MANIFIESTO_LIMPIEZA.read_text(encoding="utf-8"))
    if int(manifiesto["filas_antes"]) != len(crudo):
        raise AssertionError("El manifiesto no coincide con las filas del crudo")
    if int(manifiesto["filas_despues"]) != len(limpio):
        raise AssertionError("El manifiesto no coincide con las filas limpias")
    if int(manifiesto["atipicos"]) != len(atipicos):
        raise AssertionError("El manifiesto no coincide con los atipicos")
    if int(manifiesto["departamentos"]) != 24:
        raise AssertionError("El manifiesto no declara 24 departamentos")
    if int(manifiesto["series_observadas"]) != 216:
        raise AssertionError("El manifiesto no declara 216 series")
    if manifiesto["salidas"]["panel_precios_limpio.csv"] != sha256_archivo(LIMPIO):
        raise AssertionError("El hash del panel limpio no coincide con el manifiesto")
    if manifiesto["salidas"]["atipicos_panel.csv"] != sha256_archivo(ATIPICOS):
        raise AssertionError("El hash de atipicos no coincide con el manifiesto")

    salida = {
        "fase": "03_validacion",
        "ok": True,
        "filas_crudo": int(len(crudo)),
        "filas_limpias": int(len(limpio)),
        "atipicos": int(len(atipicos)),
        "atipicos_pct": float(100 * len(atipicos) / len(crudo)),
        "departamentos": int(len(ids)),
        "series_observadas": int(limpio.groupby(["departamento_id", "categoria"]).ngroups),
        "departamentos_limpios": int(limpio["departamento_id"].nunique()),
        "hashes": {
            "panel_precios_limpio.csv": sha256_archivo(LIMPIO),
            "atipicos_panel.csv": sha256_archivo(ATIPICOS),
        },
    }
    MANIFIESTOS.mkdir(parents=True, exist_ok=True)
    MANIFIESTO_VALIDACION.write_text(
        json.dumps(salida, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Fase 3 valida.")
    print(f"  Crudo: {len(crudo):,} | Limpio: {len(limpio):,} | Atipicos: {len(atipicos):,}")
    print(f"  Departamentos: {salida['departamentos_limpios']}")
    print(f"  Series: {salida['series_observadas']}")
    print(f"  Manifiesto: {MANIFIESTO_VALIDACION.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
