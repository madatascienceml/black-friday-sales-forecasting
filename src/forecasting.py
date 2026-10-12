"""Recursive daily forecasting shared by the notebooks and tests."""
import numpy as np
import pandas as pd

LAGS = [f"unidades_lag_{k}" for k in range(1, 8)]
MEDIA_MOVIL = "unidades_media_movil_7"


def recursive_forecast(pipeline, df_dias):
    """Forecast the units sold day by day, feeding each prediction into the next day's lags.

    The logic is the same as `simular_producto` in `app/streamlit_app.py`, but all the products
    are predicted at once on each day (one `predict` call per day instead of one per product and day).

    Parameters
    ----------
    pipeline : fitted estimator
        Any fitted object with `predict` and `feature_names_in_` (e.g. the scikit-learn Pipeline
        with the one-hot encoder and the model).
    df_dias : pandas.DataFrame
        One row per product and day, with `fecha`, `producto_id` and all the columns in
        `pipeline.feature_names_in_`. Every product must appear on every day. Only the lags
        (`unidades_lag_1..7`) and `unidades_media_movil_7` of the FIRST day are used; for the
        following days they are overwritten with values built from the predictions.

    Returns
    -------
    pandas.DataFrame
        A copy of `df_dias` sorted by `fecha` and `producto_id` (original index kept), with the
        updated lags and moving average and a new column `unidades_predichas`
        (predictions clipped to a minimum of 0).

    Notes
    -----
    On each day after the first: `unidades_lag_1` is the previous day's prediction,
    `unidades_lag_k` is the previous day's `unidades_lag_(k-1)`, and `unidades_media_movil_7`
    is the mean of the 7 updated lags.
    """
    sim = df_dias.sort_values(["fecha", "producto_id"]).copy()
    for col in LAGS + [MEDIA_MOVIL]:
        sim[col] = sim[col].astype(float)
    sim["unidades_predichas"] = 0.0

    # Every day must contain the same products in the same order
    dias = sim["fecha"].drop_duplicates().tolist()
    posiciones = [np.flatnonzero((sim["fecha"] == dia).to_numpy()) for dia in dias]
    productos = sim["producto_id"].to_numpy()[posiciones[0]]
    for pos in posiciones:
        if not np.array_equal(sim["producto_id"].to_numpy()[pos], productos):
            raise ValueError("Every day must contain the same products")

    columnas = list(pipeline.feature_names_in_)
    lags_previos = None
    pred_previa = None
    for i, pos in enumerate(posiciones):
        if i > 0:
            # lag_1 = previous prediction; lag_k = previous day's lag_(k-1)
            nuevos = np.column_stack([pred_previa, lags_previos[:, :-1]])
            sim.iloc[pos, [sim.columns.get_loc(c) for c in LAGS]] = nuevos
            # Moving average = mean of the 7 already-updated lags
            sim.iloc[pos, sim.columns.get_loc(MEDIA_MOVIL)] = nuevos.mean(axis=1)

        # Predict all the products of the day at once, clipped to a minimum of 0
        pred = np.maximum(np.asarray(pipeline.predict(sim.iloc[pos][columnas]), dtype=float), 0.0)
        sim.iloc[pos, sim.columns.get_loc("unidades_predichas")] = pred
        lags_previos = sim.iloc[pos][LAGS].to_numpy(dtype=float)
        pred_previa = pred

    return sim
