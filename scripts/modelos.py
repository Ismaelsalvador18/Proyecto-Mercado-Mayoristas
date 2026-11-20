#!/usr/bin/env python3
"""Constructores de variables y modelos compartidos.

Evita que el torneo, la representacion de modelos y la robustez definan
versiones distintas de las mismas variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import LGBM, RF, RIDGE_ALPHA, XGB

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def _mensual(d: pd.DataFrame) -> pd.DataFrame:
    idx = pd.date_range(d.index.min(), d.index.max(), freq="MS")
    return d.reindex(idx)


def construir_nowcast(panel: pd.DataFrame, cat: str) -> tuple[pd.DataFrame, pd.Series]:
    """Cambio mensual contemporaneo: dl0 de Lima explica dp de Puno."""
    d = panel[panel.categoria == cat].pivot_table(
        index="fecha", columns="region", values="precio_kg").loc["2012-01-01":, ["Puno", "Lima"]]
    d = _mensual(d)
    lp, ll = np.log(d["Puno"]), np.log(d["Lima"])
    dp, dl = lp.diff(), ll.diff()
    X = pd.DataFrame({
        "dl0": dl, "dl1": dl.shift(1), "dl2": dl.shift(2), "dl3": dl.shift(3),
        "dp1": dp.shift(1), "dp2": dp.shift(2),
    })
    t = np.arange(len(X))
    X["sin1"] = np.sin(2 * np.pi * t / 12)
    X["cos1"] = np.cos(2 * np.pi * t / 12)
    datos = pd.concat([X, dp.rename("y")], axis=1).dropna()
    return datos.drop(columns="y"), datos["y"]


def construir_forecast(panel: pd.DataFrame, cat: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Nivel del mes siguiente, con mediana de otras regiones y estacionalidad."""
    d = panel[panel.categoria == cat].pivot_table(
        index="fecha", columns="region", values="precio_kg").loc["2012-01-01":]
    otras = d.drop(columns=[c for c in ["Puno", "Lima"] if c in d.columns])
    d["otras"] = otras.median(axis=1)
    d = _mensual(d[["Puno", "Lima", "otras"]])
    lp, ll, lm = np.log(d["Puno"]), np.log(d["Lima"]), np.log(d["otras"])
    X = pd.DataFrame({"p0": lp, "p1": lp.shift(1), "l0": ll, "l1": ll.shift(1), "m0": lm})
    t = np.arange(len(X))
    X["sin1"] = np.sin(2 * np.pi * t / 12)
    X["cos1"] = np.cos(2 * np.pi * t / 12)
    y = lp.shift(-1)
    datos = pd.concat([X, y.rename("y")], axis=1).dropna()
    return datos.drop(columns="y"), datos["y"], datos["p0"]


def modelos_principales() -> dict:
    modelos = {
        "OLS": LinearRegression(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
        "Random Forest": RandomForestRegressor(**RF),
    }
    if lgb is not None:
        modelos["LightGBM"] = lgb.LGBMRegressor(**LGBM)
    return modelos


def modelos_complemento() -> dict:
    modelos = {
        "Arbol": DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=0),
        "Random Forest": RandomForestRegressor(**RF),
    }
    if lgb is not None:
        modelos["LightGBM"] = lgb.LGBMRegressor(**LGBM)
    if xgb is not None:
        modelos["XGBoost"] = xgb.XGBRegressor(**XGB)
    return modelos


def metricas(real: np.ndarray, pred: np.ndarray, base: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    real, pred, base = np.asarray(real), np.asarray(pred), np.asarray(base)
    mae = float(mean_absolute_error(real, pred))
    mae_base = float(mean_absolute_error(real, base))
    # Un cambio exactamente cero no tiene direccion: no debe contarse como fallo.
    con_direccion = real != 0
    acierto = (float(np.mean(np.sign(pred[con_direccion]) == np.sign(real[con_direccion])) * 100)
               if con_direccion.any() else np.nan)
    return {
        "mae": mae,
        "rmse": float(np.sqrt(mean_squared_error(real, pred))),
        "r2": float(r2_score(real, pred)),
        "acierto": acierto,
        "n_direccional": int(con_direccion.sum()),
        "mejora_mae": float(100 * (1 - mae / mae_base)) if mae_base > 0 else np.nan,
        "n": int(len(real)),
    }


def dm_test(error1: np.ndarray, error2: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano, perdida cuadratica, h=1, correccion HLN."""
    d = np.asarray(error1) ** 2 - np.asarray(error2) ** 2
    n = len(d)
    if n < 5:
        return np.nan, np.nan
    var = np.var(d, ddof=0)
    if var <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    return float(stat), float(2 * (1 - stats.t.cdf(abs(stat), df=n - 1)))
