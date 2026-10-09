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
* Amazon, Decathlon y Deporvillage ya no están en `df` (solo en competencia_df)
* Nombres disponibles en el notebook: `cols_onehot` (lista de las 44 columnas one-hot) y la función `aplicar_onehot(df_nuevo)`
* No uses en tu código ninguna otra variable que no esté en la lista anterior, salvo que la hayas definido tú mismo en el código que generes
* No uses ninguna librería que no sean estas: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, optuna, jupyter, ipykernel, streamlit, holidays
