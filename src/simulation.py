"""Price-plan scenarios for the November 2025 simulator."""
import pandas as pd

BLACK_FRIDAY_DAY = 28
BLACK_FRIDAY_WEEK_DAYS = range(24, 31)
VENTANAS = ("plan", "black_friday", "black_friday_week", "november")


def apply_price_plan(datos, ventana, descuento=0.0, factor_competencia=1.0):
    """Return a copy of the rows of one product with the discount applied only on the days of a window.

    Parameters
    ----------
    datos : pandas.DataFrame
        Rows of one product (or several), with `fecha`, `precio_base`, `precio_venta`,
        `descuento_porcentaje`, `precio_competencia` and `ratio_precio`.
    ventana : {"plan", "black_friday", "black_friday_week", "november"}
        Days where the discount is applied: none (the price plan of the file is kept),
        Black Friday (28 November 2025), the Black Friday week (24-30 November) or the whole month.
    descuento : float
        Discount in percent, same convention as in training (negative = discount, positive = surcharge).
        Ignored when `ventana` is "plan".
    factor_competencia : float
        Multiplier of the competitors' average price (e.g. 0.95 for -5%). It is a market scenario,
        not part of our price plan, so it applies to every day of the month whatever the window.

    Returns
    -------
    pandas.DataFrame
        Copy with `descuento_porcentaje` and `precio_venta` replaced on the window days
        (`precio_venta = precio_base * (1 + descuento / 100)`), `precio_competencia` multiplied by
        `factor_competencia`, and `ratio_precio = precio_venta / precio_competencia` recomputed on every
        row whose price or competition price changed. Untouched rows keep the values of the file;
        with "plan" and a factor of 1 the result is identical to the input.
    """
    if ventana not in VENTANAS:
        raise ValueError(f"ventana must be one of {VENTANAS}, got {ventana!r}")

    sim = datos.copy()
    dia = sim["fecha"].dt.day
    if ventana == "plan":
        en_ventana = pd.Series(False, index=sim.index)
    elif ventana == "black_friday":
        en_ventana = dia == BLACK_FRIDAY_DAY
    elif ventana == "black_friday_week":
        en_ventana = dia.isin(BLACK_FRIDAY_WEEK_DAYS)
    else:
        en_ventana = pd.Series(True, index=sim.index)

    # Discount only on the window days
    sim.loc[en_ventana, "descuento_porcentaje"] = float(descuento)
    sim.loc[en_ventana, "precio_venta"] = sim.loc[en_ventana, "precio_base"] * (1 + descuento / 100)

    # Competition scenario on the whole month
    if factor_competencia != 1.0:
        sim["precio_competencia"] = sim["precio_competencia"] * factor_competencia
        cambiadas = pd.Series(True, index=sim.index)
    else:
        cambiadas = en_ventana

    sim.loc[cambiadas, "ratio_precio"] = sim.loc[cambiadas, "precio_venta"] / sim.loc[cambiadas, "precio_competencia"]
    return sim
