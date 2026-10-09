"""Streamlit app for the DS4B project: November 2025 sales simulation.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (resolved from this file, not from the working directory)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "models" / "modelo_final.joblib"
RUTA_DATOS = RAIZ / "data" / "processed" / "inferencia_df_transformado.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MORADO = "#764ba2"
AZUL = "#667eea"
ROJO = "#e63946"
FECHA_BLACK_FRIDAY = pd.Timestamp("2025-11-28")
LAGS = [f"unidades_lag_{k}" for k in range(1, 8)]
MEDIA_MOVIL = "unidades_media_movil_7"
ESCENARIOS = {
    "Actual (0%)": 1.00,
    "Competencia -5%": 0.95,
    "Competencia +5%": 1.05,
}
TITULOS_ESCENARIO = {
    "Actual (0%)": "Sin cambios (0%)",
    "Competencia -5%": "Competencia baja (-5%)",
    "Competencia +5%": "Competencia alta (+5%)",
}
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ---------------------------------------------------------------------------
# Formatting helpers (Spanish style: 1.234,56)
# ---------------------------------------------------------------------------
def fmt_num(valor, decimales=0):
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_euros(valor, decimales=2):
    return f"{fmt_num(valor, decimales)} €"


def fmt_descuento(valor):
    # Zero is shown without sign; otherwise explicit +/-
    return "0%" if valor == 0 else f"{valor:+d}%"


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def cargar_modelo(ruta):
    return joblib.load(ruta)


@st.cache_data(show_spinner=False)
def cargar_datos(ruta):
    datos = pd.read_csv(ruta, parse_dates=["fecha"])
    return datos.sort_values(["producto_id", "fecha"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Recursive prediction (no UI, testable)
# ---------------------------------------------------------------------------
def simular_producto(modelo, datos_producto, descuento, factor_competencia):
    """Recursive 30-day forecast for one product, one discount and one scenario."""
    columnas = list(modelo.feature_names_in_)
    sim = datos_producto.sort_values("fecha").reset_index(drop=True).copy()

    # Only what depends on the controls is recalculated
    sim["descuento_porcentaje"] = float(descuento)
    sim["precio_venta"] = sim["precio_base"] * (1 + descuento / 100)
    sim["precio_competencia"] = sim["precio_competencia"] * factor_competencia
    sim["ratio_precio"] = sim["precio_venta"] / sim["precio_competencia"]

    # Float columns so the recursion can overwrite lags without dtype issues
    for col in LAGS + [MEDIA_MOVIL]:
        sim[col] = sim[col].astype(float)

    predicciones = np.zeros(len(sim))
    for i in range(len(sim)):
        if i > 0:
            # lag_1 = previous prediction; lag_k = previous day's lag_(k-1)
            previos = sim.loc[i - 1, LAGS].to_numpy(dtype=float)
            nuevos = np.concatenate(([predicciones[i - 1]], previos[:-1]))
            sim.loc[i, LAGS] = nuevos
            # Moving average = mean of the 7 already-updated lags
            sim.loc[i, MEDIA_MOVIL] = nuevos.mean()

        fila = sim.loc[[i], columnas]
        predicciones[i] = max(float(modelo.predict(fila)[0]), 0.0)

    sim["unidades_predichas"] = predicciones
    sim["ingresos_dia"] = sim["unidades_predichas"] * sim["precio_venta"]
    return sim


def resumen(sim):
    return {
        "unidades": sim["unidades_predichas"].sum(),
        "ingresos": sim["ingresos_dia"].sum(),
        "precio_medio": sim["precio_venta"].mean(),
        "descuento_medio": sim["descuento_porcentaje"].mean(),
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def grafico_unidades(sim):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    dias = sim["fecha"].dt.day
    sns.lineplot(x=dias, y=sim["unidades_predichas"], color=AZUL, linewidth=2.6, marker="o", ax=ax)

    bf = sim[sim["fecha"] == FECHA_BLACK_FRIDAY].iloc[0]
    dia_bf = FECHA_BLACK_FRIDAY.day
    ax.axvline(dia_bf, color=MORADO, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.scatter([dia_bf], [bf["unidades_predichas"]], color=ROJO, s=110, zorder=5)
    ax.annotate(
        f"Black Friday\n{fmt_num(bf['unidades_predichas'])} uds.",
        xy=(dia_bf, bf["unidades_predichas"]),
        xytext=(-85, 20),
        textcoords="offset points",
        color=MORADO,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=MORADO),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MORADO),
    )
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks(range(1, 31))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Día de noviembre de 2025")
    ax.set_ylabel("Unidades predichas")
    ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def aplicar_estilo():
    st.markdown(
        f"""
        <style>
        .cabecera {{
            background: linear-gradient(90deg, {AZUL} 0%, {MORADO} 100%);
            padding: 1.4rem 1.8rem; border-radius: 14px; color: white; margin-bottom: 1rem;
        }}
        .cabecera h1 {{ margin: 0; color: white; font-size: 2rem; }}
        .cabecera p {{ margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 1.1rem; }}
        div[data-testid="stMetric"] {{
            background: #f5f3fb; border-left: 5px solid {MORADO};
            padding: 0.8rem 1rem; border-radius: 10px;
        }}
        div[data-testid="stMetric"] * {{ color: #2d2a4a !important; }}
        section[data-testid="stSidebar"] .stButton > button {{
            background: linear-gradient(90deg, {AZUL} 0%, {MORADO} 100%);
            color: white; border: none; font-weight: 700; font-size: 1.1rem; padding: 0.7rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def estilo_tabla(tabla):
    def resaltar(fila):
        if fila["Fecha"].startswith("🛍️"):
            return [f"background-color: #e8e4fb; color: {MORADO}; font-weight: 700"] * len(fila)
        return [""] * len(fila)

    return tabla.style.apply(resaltar, axis=1)


def main():
    st.set_page_config(page_title="DS4B - Simulación de ventas", page_icon="📈", layout="wide")
    aplicar_estilo()

    # --- Loading with error handling ---
    try:
        with st.spinner("Cargando modelo y datos..."):
            modelo = cargar_modelo(RUTA_MODELO)
            datos = cargar_datos(RUTA_DATOS)
        faltan = [c for c in modelo.feature_names_in_ if c not in datos.columns]
        if faltan:
            raise ValueError(f"Faltan columnas en los datos: {faltan}")
    except FileNotFoundError as e:
        st.error(f"❌ No se encontró un archivo necesario: {e.filename or e}")
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Error al cargar el modelo o los datos: {e}")
        st.stop()

    productos = (
        datos[["producto_id", "nombre"]].drop_duplicates().sort_values("nombre").reset_index(drop=True)
    )

    # --- Sidebar ---
    with st.sidebar:
        st.title("Controles de Simulación")
        nombre = st.selectbox("📦 Selecciona el producto", productos["nombre"].tolist())
        st.divider()
        descuento = st.slider(
            "💰 Ajuste de Descuento",
            min_value=-50, max_value=50, value=0, step=5, format="%d%%",
            help="Misma convención que el entrenamiento: negativo = descuento, positivo = recargo.",
        )
        st.divider()
        escenario = st.radio(
            "🏪 Escenario de Competencia", list(ESCENARIOS.keys()),
            help="Modifica el precio medio de la competencia: -5% o +5% respecto al valor actual.",
        )
        st.divider()
        simular = st.button("🚀 Simular Ventas", width="stretch", type="primary")

    # --- Simulation (kept in session_state so widget changes don't wipe the results) ---
    if simular:
        try:
            with st.spinner("Calculando la predicción recursiva (30 días x 3 escenarios)..."):
                producto_id = productos.loc[productos["nombre"] == nombre, "producto_id"].iloc[0]
                datos_producto = datos[datos["producto_id"] == producto_id]
                resultados = {
                    nom: simular_producto(modelo, datos_producto, descuento, factor)
                    for nom, factor in ESCENARIOS.items()
                }
            st.session_state["sim"] = {
                "nombre": nombre, "descuento": descuento, "escenario": escenario, "resultados": resultados,
            }
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ Error durante la predicción: {e}")
            st.stop()

    estado = st.session_state.get("sim")
    if estado is None:
        info = datos.loc[datos["nombre"] == nombre].iloc[0]
        st.title("Dashboard de Predicción - Noviembre 2025")
        st.subheader(f"Producto: {nombre}")
        st.info("👈 Elige un producto, ajusta el descuento y pulsa **🚀 Simular Ventas** para ver la predicción.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Categoría", info["categoria"])
        c2.metric("Subcategoría", info["subcategoria"])
        c3.metric("Precio Base", fmt_euros(info["precio_base"]))
        st.divider()
        st.subheader("Vista Previa")
        st.write("Presiona 'Simular Ventas' para ver las predicciones reales")
        return

    sim = estado["resultados"][estado["escenario"]]
    r = resumen(sim)

    if (estado["nombre"], estado["descuento"], estado["escenario"]) != (nombre, descuento, escenario):
        st.info("ℹ️ Has cambiado los controles: pulsa **Simular Ventas** para actualizar los resultados.")

    # (1) Header
    st.markdown(
        f'<div class="cabecera"><h1>📈 {estado["nombre"]}</h1>'
        f'<p>Simulación noviembre 2025 · Descuento {fmt_descuento(estado["descuento"])} · {estado["escenario"]}</p></div>',
        unsafe_allow_html=True,
    )

    # (2) Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "🛒 Unidades totales", fmt_num(r["unidades"]),
        help="Suma de las unidades predichas por el modelo para los 30 días de noviembre.",
    )
    c2.metric(
        "💶 Ingresos proyectados", fmt_euros(r["ingresos"]),
        help="Suma diaria de unidades predichas por precio de venta.",
    )
    c3.metric(
        "💵 Precio promedio", fmt_euros(r["precio_medio"]),
        help="Precio de venta medio del producto durante el mes, tras aplicar el ajuste de descuento.",
    )
    c4.metric(
        "🏷️ Descuento promedio", f"{fmt_num(r['descuento_medio'], 2)} %",
        help="Descuento medio aplicado sobre el precio base (negativo = descuento, positivo = recargo).",
    )

    # Training-range and recursion warnings
    if not -15 <= estado["descuento"] <= 3:
        st.warning(
            "⚠️ Los descuentos del entrenamiento iban de -15% a +3%. Fuera de ese rango "
            "el modelo no extrapola: la respuesta de las ventas no será fiable."
        )
    st.caption(
        "🔁 La predicción es recursiva (cada día usa las predicciones anteriores como lags), "
        "por lo que los errores se acumulan a lo largo del mes."
    )

    st.divider()

    # (3) Chart
    st.subheader("📈 Predicción diaria de ventas")
    fig = grafico_unidades(sim)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # (4) Table
    st.subheader("📋 Detalle diario")
    es_bf = sim["fecha"] == FECHA_BLACK_FRIDAY
    tabla = pd.DataFrame({
        "Fecha": [("🛍️ " if bf else "") + f.strftime("%d/%m/%Y") for f, bf in zip(sim["fecha"], es_bf)],
        "Día de la semana": [DIAS_SEMANA[d] for d in sim["fecha"].dt.dayofweek],
        "Precio venta": sim["precio_venta"],
        "Precio competencia": sim["precio_competencia"],
        "Descuento (%)": sim["descuento_porcentaje"],
        "Unidades predichas": sim["unidades_predichas"],
        "Ingresos": sim["ingresos_dia"],
    })
    st.dataframe(
        # Formatting only changes the display; underlying values stay numeric for sorting
        estilo_tabla(tabla).format({
            "Precio venta": fmt_euros,
            "Precio competencia": fmt_euros,
            "Descuento (%)": lambda v: fmt_num(v, 2),
            "Unidades predichas": fmt_num,
            "Ingresos": fmt_euros,
        }),
        width="stretch", hide_index=True, height=400,
    )
    st.caption("🛍️ = Black Friday (28 de noviembre).")

    st.divider()

    # (5) Scenario comparison
    st.subheader("🔄 Comparativa de escenarios de competencia")
    st.caption(f"Manteniendo el descuento elegido ({fmt_descuento(estado['descuento'])}).")
    columnas = st.columns(len(ESCENARIOS))
    for col, (nom, sim_esc) in zip(columnas, estado["resultados"].items()):
        res = resumen(sim_esc)
        with col:
            st.markdown(f"**{TITULOS_ESCENARIO[nom]}**" + ("  ✅" if nom == estado["escenario"] else ""))
            st.metric("Unidades totales", fmt_num(res["unidades"]))
            st.metric("Ingresos", fmt_euros(res["ingresos"]))


if __name__ == "__main__":
    main()
