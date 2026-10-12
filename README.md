# Black Friday Sales Forecasting — Sports Retail

End-to-end demand forecasting project: from raw sales data to a recursive 2025 Black Friday forecast and an interactive pricing simulator built with Streamlit.

![Streamlit app](docs/images/app_dashboard.png)

## Business problem

A sports retailer needs to know how many units of each product it will sell during November, and especially on Black Friday, to plan stock and pricing. This project builds a model that forecasts daily unit sales per product for November 2025 and lets a user explore "what if" discount and competitor-price scenarios.

## Result

Validated on a full held-out year (2024), the model reaches **MAE 1.08 units/day (naive baseline: 3.67) and R² 0.87**. On Black Friday 2024 alone the error is larger (MAE 3.96, WMAPE 21.1%, R² 0.84), because peaks are systematically underestimated.

| Error by period | Black Friday 2024: real vs predicted |
|---|---|
| ![MAE by period](docs/images/mae_by_period.png) | ![Black Friday 2024](docs/images/black_friday_2024_real_vs_predicted.png) |

## Approach

- **Data:** 24 products × 37 days (25 Oct – 30 Nov) × 4 years (2021–2024) = 3,552 rows. After building lag features, 2,880 rows remain. Target: `unidades_vendidas`.
- **Features (74 predictors):** calendar features (including days to Black Friday), lags 1–7 and a 7-day moving average of sales per product and year, price variables (own price, competitor price, price ratio, discount) and one-hot encoded categories. Revenue (`ingresos`) is excluded because it is a direct function of the target.
- **Validation by year:** train on 2021–2023 (2,160 rows), validate on 2024 (720 rows). A random split would mix days of the same campaign between train and validation and give optimistic results.
- **Model:** `HistGradientBoostingRegressor`, deliberately conservative (learning rate 0.03, max depth 5, 15 leaves, `min_samples_leaf=40`, L2 regularisation) to keep the train/validation gap small. The final model is retrained on all 2,880 rows.
- **2025 forecast:** the inference file only contains sales up to 31 October, so November lags do not exist. The app forecasts **recursively**: each day's prediction feeds the lags and moving average of the next day.

## Decisions and findings

- **Lag leakage avoided.** Lags are computed per product *and* year, and the moving average uses `shift(1)` before the rolling window, so it never includes the current day's sales. A version that includes the current day leaks the target into a feature.
- **Year-based validation** reflects the real use case (predict a future year) better than a random split.
- **Discount is confounded with Black Friday.** In the training data, discounts beyond about ±3% only occur on Black Friday days (−15% / −12%). The model cannot separate the effect of the discount from the effect of the event, and trees cannot extrapolate outside the −15% … +3% range.
- **Most important features** (permutation importance): `descuento_porcentaje`, `unidades_lag_1`, `unidades_lag_7`, `unidades_media_movil_7`, `dias_hasta_black_friday`, `ratio_precio`.

![Permutation importance](docs/images/permutation_importance.png)

## Limitations

- **The simulator does not measure real price elasticity.** Because of the confounding above, a constant discount behaves more like a "Black Friday switch" than a causal price effect. Treat scenarios as model behaviour, not as a pricing recommendation.
- **Recursive forecasts accumulate error.** Validation metrics use real lags (one-step-ahead), so they are optimistic for the 2025 November forecast, where lags are themselves predictions.
- **Peaks are underestimated.** Black Friday volumes are systematically lower than real values for star products.
- **Only four training years** and 24 products: small data for a time-series problem.

## How to run

Developed and tested with Python 3.13.9.

```bash
conda create -n forecasting python=3.13
conda activate forecasting
pip install -r requirements.txt
```

> `requirements.txt` is the portable installation (only the libraries the code imports, with pinned versions). `requirements-lock.txt` is the exact environment used on Windows (full `pip freeze`, including Windows-specific packages).

1. Place the raw CSVs in `data/raw/entrenamiento/` and `data/raw/inferencia/` (data, processed files and models are git-ignored).
2. Run `notebooks/entrenamiento.ipynb` — EDA, feature engineering, validation, final model (saved to `models/modelo_final.joblib`).
3. Run `notebooks/forecasting.ipynb` — prepares the 2025 inference data.
4. Launch the app from the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

> The saved model is sensitive to the scikit-learn version: use the same version to train and to run the app.

## Project structure

```
app/            Streamlit simulator (recursive forecast)
notebooks/      entrenamiento.ipynb, forecasting.ipynb
docs/images/    Figures used in this README
data/, models/  Generated locally (git-ignored)
```

## Next steps

- Wrap preprocessing and model in a scikit-learn `Pipeline` (`ColumnTransformer` + `OneHotEncoder`).
- Hyperparameter tuning with year-based cross-validation, keeping a final year untouched.
- Compare the recursive model with a no-lag model, which avoids error accumulation.
- Fix the simulator so the slider modifies a realistic price plan instead of a constant discount.
- Move the recursion logic into a module (and possibly a FastAPI service).

## Context

Built during the free 3-day *Data Science for Business* challenge, following its base scope (Black Friday forecasting) and extending it with stricter validation, leakage checks and an analysis of the simulator's limits.
