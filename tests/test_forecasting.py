import numpy as np
import pandas as pd

from src.forecasting import LAGS, MEDIA_MOVIL, recursive_forecast


class ModeloConstante:
    """Dummy model that always predicts the same value."""

    feature_names_in_ = np.array(LAGS + [MEDIA_MOVIL])

    def __init__(self, valor):
        self.valor = valor

    def predict(self, X):
        return np.full(len(X), self.valor)


def _datos(n_productos=3, n_dias=5):
    fechas = pd.date_range("2024-11-01", periods=n_dias)
    filas = [(f, f"P{p}") for f in fechas for p in range(n_productos)]
    df = pd.DataFrame(filas, columns=["fecha", "producto_id"])
    for k, col in enumerate(LAGS, start=1):
        df[col] = float(k)
    df[MEDIA_MOVIL] = 4.0
    return df


def test_one_value_per_product_and_day():
    out = recursive_forecast(ModeloConstante(10), _datos())
    assert len(out) == 3 * 5
    assert out["unidades_predichas"].notna().all()


def test_no_negative_predictions():
    out = recursive_forecast(ModeloConstante(-5), _datos())
    assert (out["unidades_predichas"] >= 0).all()


def test_lag_1_is_previous_day_prediction():
    out = recursive_forecast(ModeloConstante(10), _datos())
    for producto, g in out.groupby("producto_id"):
        g = g.sort_values("fecha")
        # Day 1 keeps the real lags; the following days use the previous prediction
        assert g["unidades_lag_1"].iloc[0] == 1.0
        assert np.allclose(g["unidades_lag_1"].iloc[1:], g["unidades_predichas"].iloc[:-1])
        # lag_k is the previous day's lag_(k-1) and the moving average is the mean of the lags
        assert np.allclose(g["unidades_lag_2"].iloc[1:], g["unidades_lag_1"].iloc[:-1])
        assert np.allclose(g[MEDIA_MOVIL].iloc[1:], g[LAGS].iloc[1:].mean(axis=1))
