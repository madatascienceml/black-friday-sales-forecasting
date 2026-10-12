import numpy as np
import pandas as pd
import pytest

from src.simulation import apply_price_plan

VENTANAS_DIAS = {
    "black_friday": {28},
    "black_friday_week": set(range(24, 31)),
    "november": set(range(1, 31)),
}


def _producto():
    fechas = pd.date_range("2025-11-01", "2025-11-30")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"fecha": fechas, "producto_id": "P1", "precio_base": 100.0})
    df["descuento_porcentaje"] = rng.uniform(-2, 2, len(df))
    df["precio_venta"] = df["precio_base"] * (1 + df["descuento_porcentaje"] / 100)
    df["precio_competencia"] = rng.uniform(90, 110, len(df))
    df["ratio_precio"] = df["precio_venta"] / df["precio_competencia"]
    return df


@pytest.mark.parametrize("ventana", list(VENTANAS_DIAS))
def test_only_window_days_change(ventana):
    original = _producto()
    sim = apply_price_plan(original, ventana, -10, 1.0)
    en_ventana = original["fecha"].dt.day.isin(VENTANAS_DIAS[ventana])
    # Outside the window nothing changes
    pd.testing.assert_frame_equal(sim[~en_ventana], original[~en_ventana])
    # Inside the window the discount is the requested one
    assert (sim.loc[en_ventana, "descuento_porcentaje"] == -10).all()


@pytest.mark.parametrize("ventana", list(VENTANAS_DIAS))
def test_price_and_ratio_are_recomputed(ventana):
    original = _producto()
    sim = apply_price_plan(original, ventana, -10, 0.95)
    en_ventana = original["fecha"].dt.day.isin(VENTANAS_DIAS[ventana])
    assert np.allclose(sim.loc[en_ventana, "precio_venta"], 90.0)
    assert np.allclose(sim["precio_competencia"], original["precio_competencia"] * 0.95)
    assert np.allclose(sim["ratio_precio"], sim["precio_venta"] / sim["precio_competencia"])


def test_plan_returns_identical_data():
    original = _producto()
    pd.testing.assert_frame_equal(apply_price_plan(original, "plan", -30, 1.0), original)


def test_plan_keeps_prices_but_applies_competition_factor():
    original = _producto()
    sim = apply_price_plan(original, "plan", -30, 1.05)
    assert (sim["precio_venta"] == original["precio_venta"]).all()
    assert (sim["descuento_porcentaje"] == original["descuento_porcentaje"]).all()
    assert np.allclose(sim["precio_competencia"], original["precio_competencia"] * 1.05)


def test_input_is_not_modified_and_bad_window_fails():
    original = _producto()
    copia = original.copy()
    apply_price_plan(original, "november", -10, 0.95)
    pd.testing.assert_frame_equal(original, copia)
    with pytest.raises(ValueError):
        apply_price_plan(original, "week", -10)
