# Instrucciones del proyecto

* Empieza siempre tu respuesta con el emoji 🤖
* Responde siempre en español
* Las variables del dataframe `df` que tienes que usar siempre en el código que generes son:
ventas_df: ['fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria',
       'precio_base', 'es_estrella', 'unidades_vendidas', 'precio_venta',
       'ingresos']
competencia_df:['fecha', 'producto_id', 'Amazon', 'Decathlon', 'Deporvillage']
df (81 columnas): ['fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria',
       'precio_base', 'es_estrella', 'unidades_vendidas', 'precio_venta',
       'ingresos',
       'año', 'mes', 'dia_mes', 'dia_semana', 'dia_año', 'semana_año', 'trimestre',
       'es_fin_semana', 'es_festivo', 'es_vispera_festivo', 'es_black_friday',
       'es_cyber_monday', 'es_semana_black_friday', 'dias_hasta_black_friday',
       'es_inicio_mes', 'es_fin_mes',
       'unidades_lag_1', 'unidades_lag_2', 'unidades_lag_3', 'unidades_lag_4',
       'unidades_lag_5', 'unidades_lag_6', 'unidades_lag_7', 'unidades_media_movil_7',
       'descuento_porcentaje', 'precio_competencia', 'ratio_precio']
       más 44 columnas one-hot: las que empiezan por nombre_h_, categoria_h_ y subcategoria_h_ (lista completa en models/onehot_columnas.csv)
train_df y validation_df: df dividido por año (train_df 2021-2023 con 2.160 filas y validation_df 2024 con 720 filas), del notebook de entrenamiento
target ("unidades_vendidas") y predictoras: lista de las 74 predictoras (todas las columnas de df salvo fecha, ingresos, el target y las columnas de texto)
modelo (HistGradientBoostingRegressor conservador, entrenado con train_df) y modelo_final (mismos parámetros, entrenado con las 2.880 filas de df)
metricas(y_real, y_pred): función que devuelve MAE, RMSE, MAPE (%), WMAPE (%) y R2
inferencia_df (notebook notebooks/forecasting.ipynb): mismas 81 columnas que df, 720 filas (noviembre de 2025, 24 productos), con unidades_vendidas e ingresos vacíos y con los lags y la media móvil vacíos en casi todo noviembre
* Amazon, Decathlon y Deporvillage ya no están en `df` (solo en competencia_df)
* Nombres disponibles en el notebook: `cols_onehot` (lista de las 44 columnas one-hot) y la función `aplicar_onehot(df_nuevo)`
* Archivos disponibles: models/modelo_final.joblib, models/onehot_columnas.csv, data/processed/df.csv y data/processed/inferencia_df_transformado.csv
* No uses en tu código ninguna otra variable que no esté en la lista anterior, salvo que la hayas definido tú mismo en el código que generes
* No uses ninguna librería que no sean estas: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, optuna, jupyter, ipykernel, streamlit, holidays, joblib
