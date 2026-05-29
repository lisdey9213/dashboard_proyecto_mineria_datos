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
from sklearn.metrics import confusion_matrix, classification_report
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
# ENTRENAMIENTO DE MODELOS (EXACTAMENTE IGUAL QUE NOTEBOOK)
# ======================================================

print("🔧 Entrenando modelos (exactamente igual que en el notebook)...")

# --- 1. K-MEANS (igual que en notebook) ---
# Usar: Tienda, Categoría (One-Hot) + Costo_Unitario
X_cluster = pd.get_dummies(df[['Tienda', 'Categoría']], drop_first=True)
X_cluster['Costo_Unitario'] = df['Costo_Unitario']

scaler_kmeans = StandardScaler()
X_scaled = scaler_kmeans.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster_Venta'] = kmeans.fit_predict(X_scaled)

print("✓ K-Means entrenado")
print("\n--- PERFILAMIENTO DE LOS CLUSTERS ---")
print(df.groupby('Cluster_Venta')['Costo_Unitario'].mean())

# --- 2. RANDOM FOREST MULTICLASE (con ruido como en notebook) ---
X_rf = pd.get_dummies(df[['Tienda', 'Categoría']], drop_first=True)
y_rf = df['Cluster_Venta']

# Generar ruido aleatorio (10% como en notebook)
np.random.seed(42)
ruido = np.random.choice([0, 1], size=X_rf.shape, p=[0.90, 0.10])
X_rf_con_ruido = X_rf ^ ruido

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf_con_ruido, y_rf, test_size=0.2, random_state=42, stratify=y_rf
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_test_rf)

# Métricas Random Forest
acc_rf = rf_model.score(X_test_rf, y_test_rf)
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
report_rf = classification_report(y_test_rf, y_pred_rf, output_dict=True, zero_division=0)

print(f"\n✓ Random Forest entrenado (Accuracy: {acc_rf:.2%})")
print("\n--- REPORTE DE CLASIFICACIÓN (RANDOM FOREST) ---")
print(classification_report(y_test_rf, y_pred_rf))

# --- 3. XGBOOST BALANCEADO (última versión del notebook) ---
X_xgb = pd.get_dummies(df[['Tienda', 'Categoría']], drop_first=True)
y_xgb = df['Cluster_Venta']

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    reg_alpha=2,
    reg_lambda=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_xgb, y_train_xgb)
y_pred_xgb = xgb_model.predict(X_test_xgb)

acc_xgb = xgb_model.score(X_test_xgb, y_test_xgb)
cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
report_xgb = classification_report(y_test_xgb, y_pred_xgb, output_dict=True, zero_division=0)

print(f"\n✓ XGBoost Balanceado entrenado (Accuracy: {acc_xgb:.2%})")
print("\n--- REPORTE DE CLASIFICACIÓN (XGBOOST BALANCEADO) ---")
print(classification_report(y_test_xgb, y_pred_xgb))

# --- 4. Importancia de Variables (Random Forest) ---
importancias_df = pd.DataFrame({
    'Variable': X_rf.columns,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=True)

print("\n✓ Importancia de variables calculada")

# --- 5. Distribución de flores por cluster ---
dist_flores = df.groupby(['Cluster_Venta', 'Categoría']).size().reset_index(name='Frecuencia')

print("✅ Todos los modelos entrenados correctamente")

# ======================================================
# INICIAR APP
# ======================================================

app = dash.Dash(__name__)
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

    # FILTRAR
    dff = df[
        (df["Tienda"].isin(tiendas)) &
        (df["Categoría"].isin(categorias)) &
        (df["Año"].isin(anios))
    ]

    # KPIs
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

    # --- GRÁFICOS EDA ---
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
        title="📦 Distribución de Ingresos por Categoría"
    )
    fig_box.update_layout(template="plotly_dark")

    fig_scatter = px.scatter(
        dff, x="Costo_Unitario", y="Precio_Venta_Unitario",
        size="Ganancia_Neta", color="Categoría", hover_data=["Tienda"],
        title="💲 Relación Costo Unitario vs Precio de Venta"
    )
    fig_scatter.update_layout(template="plotly_dark")

    fig_pie = px.pie(
        dff.groupby("Categoría")["Ingreso_Total"].sum().reset_index(),
        names="Categoría", values="Ingreso_Total",
        title="🥧 Participación de Ventas por Categoría"
    )
    fig_pie.update_layout(template="plotly_dark")

    # Matriz de correlación
    cols_corr = ["Cantidad", "Costo_Unitario", "Costo_Total", "Precio_Venta_Unitario", "Ingreso_Total", "Ganancia_Neta"]
    corr = dff[cols_corr].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmin=-1, zmax=1
    ))
    fig_heatmap.update_layout(title="📊 Matriz de Correlación de Variables", template="plotly_dark")

    # --- GRÁFICOS ML (estáticos, iguales al notebook) ---

    # Gráfico de Clusters (K-Means)
    fig_clusters = px.scatter(
        df,
        x="Ingreso_Total", y="Ganancia_Neta",
        color=df["Cluster_Venta"].astype(str),
        size="Cantidad",
        title="🎯 Segmentación de Ventas (K-Means, k=3)",
        labels={"color": "Segmento (Cluster)", "Ingreso_Total": "Ingreso Total ($)", "Ganancia_Neta": "Ganancia Neta ($)"}
    )
    fig_clusters.update_layout(template="plotly_dark")

    # Comparación de Modelos
    fig_comparacion = px.bar(
        pd.DataFrame({"Modelo": ["Random Forest", "XGBoost"], "Accuracy": [acc_rf, acc_xgb]}),
        x="Modelo", y="Accuracy", color="Modelo", text="Accuracy",
        title="📊 Comparación de Accuracy de Modelos",
        range_y=[0, 1]
    )
    fig_comparacion.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig_comparacion.update_layout(template="plotly_dark")

    # Matriz de Confusión Random Forest
    cm_rf_df = pd.DataFrame(
        cm_rf,
        index=['Real: Cluster 0', 'Real: Cluster 1', 'Real: Cluster 2'],
        columns=['Predicho: C0', 'Predicho: C1', 'Predicho: C2']
    )
    fig_rf_confusion = go.Figure(data=go.Heatmap(
        z=cm_rf,
        x=["Predicho: C0", "Predicho: C1", "Predicho: C2"],
        y=["Real: Cluster 0", "Real: Cluster 1", "Real: Cluster 2"],
        colorscale="Blues", text=cm_rf, texttemplate="%{text}", showscale=True
    ))
    fig_rf_confusion.update_layout(
        title=f"🌲 Matriz de Confusión - Random Forest (Accuracy: {acc_rf:.2%})",
        template="plotly_dark"
    )

    # Matriz de Confusión XGBoost
    fig_xgb_confusion = go.Figure(data=go.Heatmap(
        z=cm_xgb,
        x=["Predicho: C0", "Predicho: C1", "Predicho: C2"],
        y=["Real: Cluster 0", "Real: Cluster 1", "Real: Cluster 2"],
        colorscale="Greens", text=cm_xgb, texttemplate="%{text}", showscale=True
    ))
    fig_xgb_confusion.update_layout(
        title=f"⚡ Matriz de Confusión - XGBoost (Accuracy: {acc_xgb:.2%})",
        template="plotly_dark"
    )

    # Importancia de Variables
    fig_importancias = px.bar(
        importancias_df, x="Importancia", y="Variable", orientation="h",
        title="🔍 Importancia de Variables para Determinar el Segmento de Venta",
        color="Importancia", color_continuous_scale="Viridis"
    )
    fig_importancias.update_layout(template="plotly_dark")

    # Distribución de Flores por Cluster
    fig_dist = px.bar(
        dist_flores,
        x="Cluster_Venta", y="Frecuencia", color="Categoría", barmode="group",
        title="🌺 Distribución de Tipos de Flores por Cluster de Venta",
        labels={"Cluster_Venta": "Segmento de Venta (Cluster)", "Frecuencia": "Frecuencia de Transacciones"}
    )
    fig_dist.update_layout(template="plotly_dark")

    # Métricas de Clasificación (Random Forest)
    metricas_data = []
    for i in range(3):
        cluster_key = str(i)
        if cluster_key in report_rf:
            metricas_data.append({
                "Cluster": f"Cluster {i}",
                "Precisión": report_rf[cluster_key]['precision'],
                "Recall": report_rf[cluster_key]['recall'],
                "F1-Score": report_rf[cluster_key]['f1-score']
            })
    
    metricas_df = pd.DataFrame(metricas_data)
    metricas_melt = metricas_df.melt(id_vars="Cluster", var_name="Métrica", value_name="Puntaje")
    
    fig_metricas = px.bar(
        metricas_melt, x="Cluster", y="Puntaje", color="Métrica", barmode="group",
        title="📐 Métricas de Clasificación por Segmento de Venta (Random Forest)",
        labels={"Cluster": "Segmentos de Venta (Clusters)", "Puntaje": "Puntaje (0 a 1)"},
        range_y=[0, 1.05]
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