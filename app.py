import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from xgboost import XGBClassifier
import numpy as np

# ======================================================
# CARGAR DATASET
# ======================================================

df = pd.read_csv("df_final_flores.csv")

# ======================================================
# LIMPIEZA
# ======================================================

df = df.dropna()
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Año"]   = df["Fecha"].dt.year
df["Mes"]   = df["Fecha"].dt.month_name()

# ======================================================
# PRE-ENTRENAR MODELOS ML (UNA SOLA VEZ AL INICIAR)
# ======================================================

FEATURES_ML = [
    "Cantidad",
    "Costo_Unitario",
    "Costo_Total",
    "Precio_Venta_Unitario",
    "Ingreso_Total",
    "Ganancia_Neta"
]

_df_ml = df[FEATURES_ML].dropna()

# --- Scaler y KMeans ---
_scaler = StandardScaler()
_X_scaled = _scaler.fit_transform(_df_ml)

_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
_kmeans.fit(_X_scaled)

# --- Clasificacion binaria (alta/baja ganancia) ---
_mediana = _df_ml["Ganancia_Neta"].median()
_y_bin   = (_df_ml["Ganancia_Neta"] >= _mediana).astype(int)
_X_cls   = _df_ml.drop(columns=["Ganancia_Neta"])

_X_tr, _X_te, _y_tr, _y_te = train_test_split(
    _X_cls, _y_bin, test_size=0.2, random_state=42
)

_rf = RandomForestClassifier(n_estimators=100, random_state=42)
_rf.fit(_X_tr, _y_tr)

_xgb = XGBClassifier(
    n_estimators=100, random_state=42,
    eval_metric="logloss", verbosity=0
)
_xgb.fit(_X_tr, _y_tr)

# Metricas fijas
_acc_rf  = _rf.score(_X_te, _y_te)
_acc_xgb = _xgb.score(_X_te, _y_te)
_cm_rf   = confusion_matrix(_y_te, _rf.predict(_X_te))
_cm_xgb  = confusion_matrix(_y_te, _xgb.predict(_X_te))

# --- Clasificacion multiclase por cluster ---
_y_cluster  = _kmeans.predict(_X_scaled)
_X_tr2, _X_te2, _y_tr2, _y_te2 = train_test_split(
    _df_ml[FEATURES_ML], _y_cluster, test_size=0.2, random_state=42
)
_rf_multi = RandomForestClassifier(n_estimators=100, random_state=42)
_rf_multi.fit(_X_tr2, _y_tr2)
_y_pred_multi = _rf_multi.predict(_X_te2)
_prec, _rec, _f1, _ = precision_recall_fscore_support(
    _y_te2, _y_pred_multi, labels=[0, 1, 2]
)

# ======================================================
# INICIAR APP
# ======================================================

app    = dash.Dash(__name__)
server = app.server

CARD_STYLE = {
    "padding": "20px",
    "borderRadius": "15px",
    "color": "white",
    "width": "23%",
    "textAlign": "center",
    "boxShadow": "2px 2px 10px rgba(0,0,0,0.2)"
}

# ======================================================
# LAYOUT
# ======================================================

app.layout = html.Div([

    html.H1(
        "🌸 Dashboard Ejecutivo - Florerías",
        style={"textAlign": "center", "color": "white", "marginBottom": "10px"}
    ),

    html.P(
        "Análisis interactivo de ventas, ingresos y rentabilidad de las diferentes categorías de flores.",
        style={"textAlign": "center", "color": "white", "fontSize": "18px", "marginBottom": "30px"}
    ),

    # FILTROS
    html.Div([

        html.Div([
            html.Label("Seleccionar Tienda", style={"color": "white"}),
            dcc.Dropdown(
                id="filtro_tienda",
                options=[{"label": i, "value": i} for i in df["Tienda"].unique()],
                value=df["Tienda"].unique().tolist(),
                multi=True
            )
        ], style={"width": "32%"}),

        html.Div([
            html.Label("Seleccionar Categoría", style={"color": "white"}),
            dcc.Dropdown(
                id="filtro_categoria",
                options=[{"label": i, "value": i} for i in df["Categoría"].unique()],
                value=df["Categoría"].unique().tolist(),
                multi=True
            )
        ], style={"width": "32%"}),

        html.Div([
            html.Label("Seleccionar Año", style={"color": "white"}),
            dcc.Dropdown(
                id="filtro_anio",
                options=[{"label": i, "value": i} for i in sorted(df["Año"].unique())],
                value=sorted(df["Año"].unique()),
                multi=True
            )
        ], style={"width": "32%"})

    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "30px"}),

    # KPIs
    html.Div(id="kpis", style={"display": "flex", "justifyContent": "space-between", "marginBottom": "40px"}),

    # GRAFICOS EDA
    dcc.Graph(id="linea_ventas"),

    html.Div([
        html.Div([dcc.Graph(id="bar_categoria")], style={"width": "49%"}),
        html.Div([dcc.Graph(id="bar_tienda")],    style={"width": "49%"})
    ], style={"display": "flex", "justifyContent": "space-between"}),

    html.Div([
        html.Div([dcc.Graph(id="boxplot")], style={"width": "49%"}),
        html.Div([dcc.Graph(id="scatter")], style={"width": "49%"})
    ], style={"display": "flex", "justifyContent": "space-between"}),

    html.Div([
        html.Div([dcc.Graph(id="pie")],     style={"width": "49%"}),
        html.Div([dcc.Graph(id="heatmap")], style={"width": "49%"})
    ], style={"display": "flex", "justifyContent": "space-between"}),

    # SECCION ML
    html.H2(
        "🤖 Modelos de Machine Learning",
        style={"color": "white", "marginTop": "40px", "textAlign": "center"}
    ),

    html.Div([
        html.Div([dcc.Graph(id="clusters")],           style={"width": "49%"}),
        html.Div([dcc.Graph(id="comparacion_modelos")], style={"width": "49%"})
    ], style={"display": "flex", "justifyContent": "space-between"}),

    html.Div([
        html.Div([dcc.Graph(id="rf_confusion")],  style={"width": "49%"}),
        html.Div([dcc.Graph(id="xgb_confusion")], style={"width": "49%"})
    ], style={"display": "flex", "justifyContent": "space-between"}),

    dcc.Graph(id="importancias"),

    html.Div([
        html.Div([dcc.Graph(id="dist_flores_cluster")],  style={"width": "49%"}),
        html.Div([dcc.Graph(id="metricas_clasificacion")], style={"width": "49%"})
    ], style={"display": "flex", "justifyContent": "space-between"})

], style={"backgroundColor": "#111111", "padding": "30px", "fontFamily": "Arial"})

# ======================================================
# CALLBACK
# ======================================================

@app.callback(
    [
        Output("kpis",                  "children"),
        Output("linea_ventas",          "figure"),
        Output("bar_categoria",         "figure"),
        Output("bar_tienda",            "figure"),
        Output("boxplot",               "figure"),
        Output("scatter",               "figure"),
        Output("pie",                   "figure"),
        Output("heatmap",               "figure"),
        Output("clusters",              "figure"),
        Output("comparacion_modelos",   "figure"),
        Output("rf_confusion",          "figure"),
        Output("xgb_confusion",         "figure"),
        Output("importancias",          "figure"),
        Output("dist_flores_cluster",   "figure"),
        Output("metricas_clasificacion","figure")
    ],
    [
        Input("filtro_tienda",    "value"),
        Input("filtro_categoria", "value"),
        Input("filtro_anio",      "value")
    ]
)
def actualizar_dashboard(tiendas, categorias, anios):

    # FILTRAR — proteger listas None o vacías
    tiendas    = tiendas    or df["Tienda"].unique().tolist()
    categorias = categorias or df["Categoría"].unique().tolist()
    anios      = anios      or sorted(df["Año"].unique())

    dff = df[
        (df["Tienda"].isin(tiendas)) &
        (df["Categoría"].isin(categorias)) &
        (df["Año"].isin(anios))
    ]

    # Si no hay datos tras el filtro, usar el dataset completo
    if dff.empty:
        dff = df.copy()

    # --------------------------------------------------
    # KPIs
    # --------------------------------------------------
    ingresos  = dff["Ingreso_Total"].sum()
    ganancias = dff["Ganancia_Neta"].sum()
    unidades  = dff["Cantidad"].sum()
    ticket    = dff["Ingreso_Total"].mean()

    kpis = [
        html.Div([html.H3("💰 Ingresos"),       html.H1(f"${ingresos:,.0f}")],  style={**CARD_STYLE, "backgroundColor": "#16A085"}),
        html.Div([html.H3("📈 Ganancia"),        html.H1(f"${ganancias:,.0f}")], style={**CARD_STYLE, "backgroundColor": "#2980B9"}),
        html.Div([html.H3("🛒 Unidades"),        html.H1(f"{unidades:,.0f}")],   style={**CARD_STYLE, "backgroundColor": "#8E44AD"}),
        html.Div([html.H3("🧾 Ticket Promedio"), html.H1(f"${ticket:,.0f}")],    style={**CARD_STYLE, "backgroundColor": "#D35400"})
    ]

    # --------------------------------------------------
    # EDA CHARTS  (solo operaciones sobre dff filtrado)
    # --------------------------------------------------
    fig_linea = px.line(
        dff.groupby("Fecha")["Ingreso_Total"].sum().reset_index(),
        x="Fecha", y="Ingreso_Total", title="📈 Evolución Temporal de Ventas"
    )
    fig_linea.update_layout(template="plotly_dark")

    fig_categoria = px.bar(
        dff.groupby("Categoría")["Ganancia_Neta"].sum().reset_index(),
        x="Categoría", y="Ganancia_Neta", color="Categoría",
        title="🌸 Ganancia por Categoría"
    )
    fig_categoria.update_layout(template="plotly_dark")

    fig_tienda = px.bar(
        dff.groupby("Tienda")["Ingreso_Total"].sum().reset_index(),
        x="Tienda", y="Ingreso_Total", color="Tienda",
        title="🏪 Ingresos por Tienda"
    )
    fig_tienda.update_layout(template="plotly_dark")

    fig_box = px.box(
        dff, x="Categoría", y="Ingreso_Total", color="Categoría",
        title="📦 Distribución de Ingresos"
    )
    fig_box.update_layout(template="plotly_dark")

    fig_scatter = px.scatter(
        dff, x="Costo_Unitario", y="Precio_Venta_Unitario",
        size="Ganancia_Neta", color="Categoría", hover_data=["Tienda"],
        title="💲 Relación Costos vs Precio"
    )
    fig_scatter.update_layout(template="plotly_dark")

    fig_pie = px.pie(
        dff.groupby("Categoría")["Ingreso_Total"].sum().reset_index(),
        names="Categoría", values="Ingreso_Total",
        title="🥧 Participación de Ventas"
    )
    fig_pie.update_layout(template="plotly_dark")

    columnas = ["Cantidad","Costo_Unitario","Costo_Total","Precio_Venta_Unitario","Ingreso_Total","Ganancia_Neta"]
    corr = dff[columnas].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmin=-1, zmax=1
    ))
    fig_heatmap.update_layout(title="🔥 Matriz de Correlación", template="plotly_dark")

    # --------------------------------------------------
    # ML — solo visualizaciones con modelos pre-entrenados
    # --------------------------------------------------

    # Clusters scatter (filtrado)
    dff_ml_filt = dff[FEATURES_ML].dropna().copy()

    if len(dff_ml_filt) > 0:
        dff_ml_filt["Cluster"] = _kmeans.predict(
            _scaler.transform(dff_ml_filt)
        ).astype(str)
    else:
        dff_ml_filt["Cluster"] = "0"

    fig_clusters = px.scatter(
        dff_ml_filt, x="Ingreso_Total", y="Ganancia_Neta",
        color="Cluster", size="Cantidad",
        title="🔵 Clustering K-Means (k=3)",
        labels={"Cluster": "Segmento"}
    )
    fig_clusters.update_layout(template="plotly_dark")

    # Comparacion modelos (valores fijos del entrenamiento global)
    fig_comparacion = px.bar(
        pd.DataFrame({"Modelo": ["Random Forest", "XGBoost"], "Accuracy": [_acc_rf, _acc_xgb]}),
        x="Modelo", y="Accuracy", color="Modelo", text="Accuracy",
        title="📊 Comparación de Modelos", range_y=[0, 1]
    )
    fig_comparacion.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig_comparacion.update_layout(template="plotly_dark")

    # Confusion matrices (pre-calculadas)
    fig_rf_confusion = go.Figure(data=go.Heatmap(
        z=_cm_rf, x=["Pred: Baja","Pred: Alta"], y=["Real: Baja","Real: Alta"],
        colorscale="Blues", text=_cm_rf, texttemplate="%{text}", showscale=True
    ))
    fig_rf_confusion.update_layout(
        title=f"🌲 Confusion Matrix - Random Forest (Acc: {_acc_rf:.2%})",
        template="plotly_dark"
    )

    fig_xgb_confusion = go.Figure(data=go.Heatmap(
        z=_cm_xgb, x=["Pred: Baja","Pred: Alta"], y=["Real: Baja","Real: Alta"],
        colorscale="Greens", text=_cm_xgb, texttemplate="%{text}", showscale=True
    ))
    fig_xgb_confusion.update_layout(
        title=f"⚡ Confusion Matrix - XGBoost (Acc: {_acc_xgb:.2%})",
        template="plotly_dark"
    )

    # Importancias (pre-calculadas)
    importancias_df = pd.DataFrame({
        "Variable": _X_cls.columns,
        "Importancia": _rf.feature_importances_
    }).sort_values("Importancia", ascending=True)

    fig_importancias = px.bar(
        importancias_df, x="Importancia", y="Variable", orientation="h",
        title="🔍 Importancia de Variables - Random Forest",
        color="Importancia", color_continuous_scale="Viridis"
    )
    fig_importancias.update_layout(template="plotly_dark")

    # Distribucion flores por cluster (filtrado)
    dff_cat = dff[FEATURES_ML + ["Categoría"]].dropna().copy()

    if len(dff_cat) > 0:
        dff_cat["Cluster"] = "Cluster " + _kmeans.predict(
            _scaler.transform(dff_cat[FEATURES_ML])
        ).astype(str)
    else:
        dff_cat["Cluster"] = "Cluster 0"

    fig_dist = px.bar(
        dff_cat.groupby(["Cluster","Categoría"]).size().reset_index(name="Frecuencia"),
        x="Cluster", y="Frecuencia", color="Categoría", barmode="group",
        title="🌺 Distribución de Tipos de Flores por Cluster de Venta",
        labels={"Cluster": "Segmento de Venta (Cluster)", "Frecuencia": "Frecuencia de Transacciones"}
    )
    fig_dist.update_layout(template="plotly_dark")

    # Metricas clasificacion (pre-calculadas)
    metricas_melt = pd.DataFrame({
        "Cluster":    ["Cluster 0","Cluster 1","Cluster 2"],
        "precision":  _prec,
        "recall":     _rec,
        "f1-score":   _f1
    }).melt(id_vars="Cluster", var_name="Métrica", value_name="Puntaje")

    fig_metricas = px.bar(
        metricas_melt, x="Cluster", y="Puntaje", color="Métrica", barmode="group",
        title="📐 Métricas de Clasificación por Segmento de Venta",
        labels={"Cluster": "Segmentos de Venta (Clusters)", "Puntaje": "Puntaje (Escala 0 a 1)"},
        range_y=[0, 1.05],
        color_discrete_map={"precision": "#2C3E6B", "recall": "#E07B2A", "f1-score": "#2ECC71"}
    )
    fig_metricas.update_layout(template="plotly_dark")

    return (
        kpis, fig_linea, fig_categoria, fig_tienda,
        fig_box, fig_scatter, fig_pie, fig_heatmap,
        fig_clusters, fig_comparacion,
        fig_rf_confusion, fig_xgb_confusion,
        fig_importancias, fig_dist, fig_metricas
    )


# ======================================================
# EJECUTAR
# ======================================================

if __name__ == "__main__":
    app.run(debug=True, port=8052)
