#!/usr/bin/env python3
"""Catalogo unico de los 24 departamentos del Peru.

La fuente de verdad es ``data/reference/departamentos.csv``. Este modulo la
carga y falla de forma explicita si el archivo no cumple el contrato:

  - exactamente 24 filas;
  - identificadores ISO y codigos SISAP unicos;
  - nombres oficiales y alias ASCII unicos;
  - coordenadas y zona natural completas;
  - una sola referencia Lima;
  - ningun mercado provincial ni Callao.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CATALOGO = RAIZ / "data" / "reference" / "departamentos.csv"

COLUMNAS = [
    "departamento_id",
    "departamento",
    "departamento_ascii",
    "codigo_sisap",
    "latitud",
    "longitud",
    "zona_natural",
    "es_referencia_lima",
]

# Los 24 departamentos oficiales del Peru. El catalogo debe coincidir con
# este conjunto; la lista se usa como contraste, no como fuente paralela.
DEPARTAMENTOS_OFICIALES = {
    "PE-AMA": "Amazonas",
    "PE-ANC": "Ancash",
    "PE-APU": "Apurimac",
    "PE-ARE": "Arequipa",
    "PE-AYA": "Ayacucho",
    "PE-CAJ": "Cajamarca",
    "PE-CUS": "Cusco",
    "PE-HUV": "Huancavelica",
    "PE-HUC": "Huanuco",
    "PE-ICA": "Ica",
    "PE-JUN": "Junin",
    "PE-LAL": "La Libertad",
    "PE-LAM": "Lambayeque",
    "PE-LIM": "Lima",
    "PE-LOR": "Loreto",
    "PE-MDD": "Madre de Dios",
    "PE-MOQ": "Moquegua",
    "PE-PAS": "Pasco",
    "PE-PIU": "Piura",
    "PE-PUN": "Puno",
    "PE-SAM": "San Martin",
    "PE-TAC": "Tacna",
    "PE-TUM": "Tumbes",
    "PE-UCA": "Ucayali",
}

# Mercados que SISAP reporta aparte y que no son departamentos.
EXCLUIDOS = {
    "030201": ("Andahuaylas", "provincia de Apurimac"),
    "060401": ("Chota", "provincia de Cajamarca"),
    "060801": ("Jaen", "provincia de Cajamarca"),
    "070000": ("Callao", "provincia constitucional"),
}

ZONAS_VALIDAS = {"Costa", "Sierra", "Selva"}


def cargar_catalogo(ruta: Path | str = CATALOGO) -> pd.DataFrame:
    """Carga y valida el catalogo departamental."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el catalogo departamental: {ruta}")

    df = pd.read_csv(
        ruta,
        dtype={
            "departamento_id": str,
            "departamento": str,
            "departamento_ascii": str,
            "codigo_sisap": str,
            "zona_natural": str,
        },
    )

    faltantes = [col for col in COLUMNAS if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el catalogo: {faltantes}")

    df = df[COLUMNAS].copy()
    df["departamento_id"] = df["departamento_id"].str.strip()
    df["departamento"] = df["departamento"].str.strip()
    df["departamento_ascii"] = df["departamento_ascii"].str.strip()
    df["codigo_sisap"] = df["codigo_sisap"].str.zfill(6)
    df["zona_natural"] = df["zona_natural"].str.strip()
    df["latitud"] = pd.to_numeric(df["latitud"], errors="raise")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="raise")
    df["es_referencia_lima"] = pd.to_numeric(
        df["es_referencia_lima"], errors="raise"
    ).astype(int)

    if len(df) != 24:
        raise ValueError(f"El catalogo debe tener 24 departamentos; tiene {len(df)}")
    if df.isnull().any().any():
        columnas = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"El catalogo tiene nulos en: {columnas}")

    for columna in ["departamento_id", "departamento", "departamento_ascii", "codigo_sisap"]:
        duplicados = df[columna][df[columna].duplicated()].tolist()
        if duplicados:
            raise ValueError(f"Valores duplicados en {columna}: {duplicados}")

    ids = set(df["departamento_id"])
    esperados = set(DEPARTAMENTOS_OFICIALES)
    if ids != esperados:
        faltan = sorted(esperados - ids)
        sobran = sorted(ids - esperados)
        raise ValueError(f"Catalogo distinto de los 24 oficiales. Faltan={faltan}; sobran={sobran}")

    if set(df["zona_natural"]) - ZONAS_VALIDAS:
        raise ValueError(f"Zonas invalidas: {sorted(set(df['zona_natural']) - ZONAS_VALIDAS)}")
    if not df["latitud"].between(-20, 0).all() or not df["longitud"].between(-82, -68).all():
        raise ValueError("Coordenadas fuera del rango del Peru")
    if int(df["es_referencia_lima"].sum()) != 1:
        raise ValueError("Debe haber exactamente una referencia Lima")
    if df.loc[df["es_referencia_lima"] == 1, "departamento_id"].iloc[0] != "PE-LIM":
        raise ValueError("La referencia debe ser PE-LIM")

    codigos_excluidos = set(EXCLUIDOS)
    if codigos_excluidos & set(df["codigo_sisap"]):
        raise ValueError("El catalogo contiene codigos de mercados excluidos")
    nombres_excluidos = {nombre.casefold() for nombre, _ in EXCLUIDOS.values()}
    if nombres_excluidos & set(df["departamento_ascii"].str.casefold()):
        raise ValueError("El catalogo contiene nombres de mercados excluidos")

    return df.sort_values("departamento_id").reset_index(drop=True)


def validar_departamentos(
    datos: pd.DataFrame,
    columna: str = "departamento_id",
    catalogo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exige que un DataFrame use exactamente los 24 departamentos oficiales."""
    catalogo = cargar_catalogo() if catalogo is None else catalogo
    if columna not in datos.columns:
        raise ValueError(f"Falta la columna {columna!r} para validar departamentos")

    observados = set(datos[columna].dropna().astype(str))
    esperados = set(catalogo["departamento_id"])
    faltan = sorted(esperados - observados)
    sobran = sorted(observados - esperados)
    if faltan or sobran:
        raise ValueError(f"Departamentos incompletos o no oficiales. Faltan={faltan}; sobran={sobran}")
    return catalogo


def mapa_codigos(catalogo: pd.DataFrame | None = None) -> dict[str, dict[str, str]]:
    """Devuelve el mapeo codigo SISAP -> metadatos canonicos."""
    catalogo = cargar_catalogo() if catalogo is None else catalogo
    return {
        fila.codigo_sisap: {
            "departamento_id": fila.departamento_id,
            "departamento": fila.departamento,
            "departamento_ascii": fila.departamento_ascii,
            "zona_natural": fila.zona_natural,
        }
        for fila in catalogo.itertuples(index=False)
    }
