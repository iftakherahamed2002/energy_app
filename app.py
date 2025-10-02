# -*- coding: utf-8 -*-
"""
Construction Energy & CO₂ Dashboard – KUET Textile Thesis
Author: Iftakher Ahamed

Updated: Fast/Full K-Fold CV (button-triggered), non-blocking UX
"""

import os, io, json, base64, warnings
from datetime import datetime

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

# ML – classic
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR as SKSVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# DL – optional (guard import)
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except Exception:
    TF_AVAILABLE = False

warnings.filterwarnings("ignore")

# =========================
# Constants / Utility
# =========================
REQUIRED_FEATURES = [
    "Wind Speed (m/s)",
    "Relative Humidity (%)",
    "Outdoor Temperature (°C)",
    "Indoor Temperature (°C)",
    "Lighting (LUX)",
    "Solar Radiation (kw/m²)",
]
TARGET_COL = "Energy Consumption (kWh)"

def kwh_to_tonnes(kwh, ef_kg_per_kwh):  # kg → t
    return (kwh * ef_kg_per_kwh) / 1000.0

def ensure_dir(path="./exports"):
    os.makedirs(path, exist_ok=True); return path

def parse_contents(contents, filename, required_cols=None):
    if contents is None: return None, "No file."
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(decoded))
        else:
            return None, f"Unsupported file type: {filename}"
    except Exception as e:
        return None, f"Failed to read {filename}: {e}"

    df.columns = [c.replace("  ", " ").strip() for c in df.columns]
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Missing columns in {filename}: {missing}"
    return df, None

def scenario_energies(E_base, share_light, share_hvac, sav_light, sav_hvac, pv_offset):
    E_light  = E_base * (1 - share_light * sav_light)
    E_hvac   = E_base * (1 - share_hvac  * sav_hvac)
    E_hybrid = E_base * (1 - share_light * sav_light) * (1 - share_hvac * sav_hvac)
    E_pv     = E_base * (1 - pv_offset)
    return {
        "Base Case": E_base,
        "Efficient Lighting": E_light,
        "Optimized HVAC": E_hvac,
        "Hybrid (Lighting+HVAC)": E_hybrid,
        "Rooftop PV (15%)": E_pv
    }

def make_result_table(scen_kwh, ef, embodied_tonnes):
    labels = list(scen_kwh.keys())
    operational_t = {k: kwh_to_tonnes(v, ef) for k, v in scen_kwh.items()}
    total_t       = {k: embodied_tonnes + operational_t[k] for k in labels}
    base_total    = total_t["Base Case"]
    reduction_pct = {k: (1 - total_t[k] / base_total) * 100 for k in labels}
    return pd.DataFrame({
        "Scenario": labels,
        "Embodied_CO2 (t)": [embodied_tonnes]*len(labels),
        "Operational_CO2 (t)": [operational_t[k] for k in labels],
        "Total_CO2 (t)": [total_t[k] for k in labels],
        "Reduction_vs_Base (%)": [reduction_pct[k] for k in labels],
        "Energy (kWh)": [scen_kwh[k] for k in labels],
    })

def make_horizontal_bar(df):
    fig = go.Figure()
    fig.add_bar(y=df["Scenario"], x=df["Embodied_CO2 (t)"], orientation='h', name="Embodied",
                marker_color="#1f77b4", text=[f"{v:.1f}" for v in df["Embodied_CO2 (t)"]],
                textposition="outside")
    fig.add_bar(y=df["Scenario"], x=df["Operational_CO2 (t)"], orientation='h', name="Operational",
                marker_color="#ff7f0e", text=[f"{v:.1f}" for v in df["Operational_CO2 (t)"]],
                textposition="outside")
    fig.add_bar(y=df["Scenario"], x=df["Total_CO2 (t)"], orientation='h', name="Total",
                marker_color="#2ca02c", text=[f"{v:.1f}" for v in df["Total_CO2 (t)"]],
                textposition="outside")
    fig.update_layout(barmode="group", title="Embodied vs Operational vs Total CO₂",
                      xaxis_title="CO₂ Emissions (tonnes)",
                      yaxis=dict(autorange="reversed"),
                      legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
                      margin=dict(l=110, r=40, t=60, b=90), height=520)
    mt = float(np.nanmax(df["Total_CO2 (t)"])) if len(df) else 0
    fig.update_xaxes(range=[0, mt*1.2 if mt>0 else 10])
    return fig

def make_reduction_bar(df):
    fig = px.bar(df, x="Scenario", y="Reduction_vs_Base (%)",
                 title="Reduction vs Base (%) – Total CO₂")
    fig.update_layout(yaxis_title="Reduction (%)", margin=dict(l=40, r=20, t=60, b=40), height=420)
    return fig

def make_time_series(sim_df=None, real_df=None):
    fig = go.Figure()
    if sim_df is not None and TARGET_COL in sim_df.columns:
        x = np.arange(1, len(sim_df)+1)
        fig.add_scatter(x=x, y=sim_df[TARGET_COL], mode="lines", name="Simulation (kWh)",
                        line=dict(dash="dash"))
    if real_df is not None and TARGET_COL in real_df.columns:
        x = np.arange(1, len(real_df)+1)
        fig.add_scatter(x=x, y=real_df[TARGET_COL], mode="lines+markers", name="Real/Sensor (kWh)")
    title = "Energy Time Series (Simulation vs Real)" if fig.data else \
            "Upload a dataset with ‘Energy Consumption (kWh)’"
    fig.update_layout(title=title, xaxis_title="Hour Index", yaxis_title="Energy (kWh)",
                      margin=dict(l=40, r=20, t=60, b=40), height=420)
    return fig

# =========================
# App
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], title="Construction Energy & CO₂ – KUET Textile Thesis")
server = app.server

banner = dbc.Alert([
    html.H3("Construction Energy & CO₂ Dashboard — KUET Textile Thesis", className="mb-1"),
    html.Div("Interactive scenario analysis + fast K-Fold ML benchmarking (button-triggered).", className="mb-0"),
], color="info", className="mb-3")

upload_card = dbc.Card(dbc.CardBody([
    html.H5("Upload Data", className="card-title"),
    html.Small("Simulation (DesignBuilder) is required; Real/Sensor optional.", className="text-muted"),
    html.Ul([html.Li(html.Code(TARGET_COL)), html.Li(", ".join(REQUIRED_FEATURES))], className="mb-2"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Simulation Excel/CSV"),
            dcc.Upload(id="upload_sim",
                children=html.Div(['Drag & Drop or ', html.A('Select Simulation File')]),
                multiple=False, style={"width":"100%","height":"70px","lineHeight":"70px",
                "borderWidth":"1px","borderStyle":"dashed","borderRadius":"6px","textAlign":"center"}),
            html.Div(id="sim_info", className="mt-2")
        ], md=6),
        dbc.Col([
            dbc.Label("Real/Sensor Excel/CSV (optional)"),
            dcc.Upload(id="upload_real",
                children=html.Div(['Drag & Drop or ', html.A('Select Real File')]),
                multiple=False, style={"width":"100%","height":"70px","lineHeight":"70px",
                "borderWidth":"1px","borderStyle":"dashed","borderRadius":"6px","textAlign":"center"}),
            html.Div(id="real_info", className="mt-2")
        ], md=6),
    ], className="g-3"),
    html.Hr(className="my-2"),
    html.Div(id="preview_tables")
]))

controls_card = dbc.Card(dbc.CardBody([
    html.H5("Parameters", className="card-title"),
    dbc.Row([dbc.Col([
        dbc.Label("Grid Emission Factor (kg CO₂/kWh)"),
        dcc.Slider(id="ef", min=0.30, max=1.20, step=0.01, value=0.70)
    ], width=12)], className="mb-2"),
    dbc.Row([
        dbc.Col([dbc.Label("Lighting share of load"), dcc.Slider(id="share_light", min=0, max=0.8, step=0.01, value=0.25)], md=6),
        dbc.Col([dbc.Label("HVAC share of load"),     dcc.Slider(id="share_hvac",  min=0, max=0.9, step=0.01, value=0.50)], md=6),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col([dbc.Label("Lighting saving (%)"), dcc.Slider(id="sav_light", min=0, max=0.5, step=0.01, value=0.20)], md=6),
        dbc.Col([dbc.Label("HVAC saving (%)"),     dcc.Slider(id="sav_hvac",  min=0, max=0.5, step=0.01, value=0.15)], md=6),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col([dbc.Label("PV offset (% total)"), dcc.Slider(id="pv_offset", min=0, max=0.5, step=0.01, value=0.15)], md=6),
        dbc.Col([dbc.Label("Embodied CO₂ (t)"),    dcc.Slider(id="embodied",  min=0, max=500, step=0.1, value=137.7)], md=6),
    ], className="mb-2"),
    html.Small("Shares are applied to base energy; savings reduce those shares.", className="text-muted"),
    html.Hr(),
    dbc.Button("Save Results", id="save_btn", color="primary", className="me-2"),
    dbc.Button("Export CSV", id="dl_csv_btn", color="secondary", className="me-2"),
    dbc.Button("Export JSON", id="dl_json_btn", color="secondary"),
    dcc.Download(id="dl_csv"), dcc.Download(id="dl_json"),
    html.Div(id="save_msg", className="mt-2 text-success")
]))

# ---- K-Fold control panel (new) ----
cv_controls = dbc.Card(dbc.CardBody([
    html.H5("ML Benchmark (K-Fold) – Quick vs Full", className="card-title"),
    html.Small("Runs on Simulation dataset. Use Quick mode during demo; Full mode for thesis figures.", className="text-muted"),
    dbc.Row([
        dbc.Col([dbc.Label("Speed Mode"),
                 dcc.Dropdown(id="cv_mode",
                    options=[{"label":"Fast (demo)","value":"fast"},
                             {"label":"Full (accurate)","value":"full"}],
                    value="fast", clearable=False)], md=4),
        dbc.Col([dbc.Label("Folds"), dcc.Input(id="cv_folds", type="number", min=2, max=10, step=1, value=3)], md=2),
        dbc.Col([dbc.Label("Sample Limit (rows)"),
                 dcc.Input(id="cv_limit", type="number", min=500, step=500, value=2000)], md=3),
        dbc.Col([dbc.Label("Include DL (GRU+Transformer)"),
                 dcc.Checklist(id="cv_dl", options=[{"label":" Enable","value":"dl"}],
                               value=[], inline=True)], md=3),
    ], className="mb-2"),
    dbc.Button("Run 5-Fold CV", id="run_cv_btn", color="success"),
    html.Span(id="cv_status", className="ms-3 text-muted"),
    html.Hr(className="my-2"),
    dcc.Graph(id="cv_bar_metrics"),               # metrics bar (MAE/RMSE/R2) per model
    dash_table.DataTable(
        id="cv_overview_table",
        columns=[{"name":"Model","id":"model"},{"name":"MAE","id":"mae"},
                 {"name":"RMSE","id":"rmse"},{"name":"R²","id":"r2"}],
        data=[], page_size=10, style_table={"overflowX":"auto"},
        style_cell={"minWidth":110}
    )
]))

graphs_card = dbc.Card(dbc.CardBody([
    html.H5("Visualizations", className="card-title"),
    dcc.Tabs([
        dcc.Tab(label="CO₂ Scenarios (Embodied, Operational, Total)", children=[
            dcc.Graph(id="stacked_bar"),
            dcc.Graph(id="reduction_bar")
        ]),
        dcc.Tab(label="Energy Time Series", children=[dcc.Graph(id="timeseries")]),
        dcc.Tab(label="ML Comparison (XGBoost)", children=[
            html.Small("Trains XGBoost on Simulation features and predicts energy from Real/Sensor features.", className="text-muted"),
            dcc.Graph(id="ml_compare"),
            dash_table.DataTable(id="ml_metrics",
                columns=[{"name":"METRIC","id":"metric"},{"name":"VALUE","id":"value"}],
                data=[], style_table={"overflowX":"auto"}, page_size=5)
        ]),
        dcc.Tab(label="ML Benchmark (5-Fold CV)", children=[cv_controls]),
    ])
]))

table_card = dbc.Card(dbc.CardBody([
    html.H5("Scenario Results Table", className="card-title"),
    dash_table.DataTable(
        id="result_table",
        columns=[
            {"name":"Scenario","id":"Scenario"},
            {"name":"Embodied_CO2 (t)","id":"Embodied_CO2 (t)","type":"numeric","format":{"specifier":".2f"}},
            {"name":"Operational_CO2 (t)","id":"Operational_CO2 (t)","type":"numeric","format":{"specifier":".2f"}},
            {"name":"Total_CO2 (t)","id":"Total_CO2 (t)","type":"numeric","format":{"specifier":".2f"}},
            {"name":"Reduction_vs_Base (%)","id":"Reduction_vs_Base (%)","type":"numeric","format":{"specifier":".1f"}},
            {"name":"Energy (kWh)","id":"Energy (kWh)","type":"numeric","format":{"specifier":".0f"}}
        ],
        data=[], style_table={"overflowX":"auto"}, page_size=10
    )
]))

stores = html.Div([dcc.Store(id="store_sim_df"), dcc.Store(id="store_real_df"), dcc.Store(id="store_result_df")])
footer = html.Div(className="text-muted mt-3", children="Developer : Iftakher Ahamed")

app.layout = dbc.Container([
    banner, stores,
    dbc.Row([dbc.Col(upload_card, lg=7), dbc.Col(controls_card, lg=5)], className="mb-3"),
    dbc.Row([dbc.Col(graphs_card, lg=7), dbc.Col(table_card,  lg=5)]),
    footer
], fluid=True)

# =========================
# Callbacks – Upload & Basic Charts
# =========================
@app.callback(
    Output("store_sim_df","data"), Output("sim_info","children"),
    Input("upload_sim","contents"), State("upload_sim","filename"), prevent_initial_call=True)
def handle_upload_sim(contents, filename):
    if not contents: return no_update, no_update
    df, err = parse_contents(contents, filename, required_cols=[TARGET_COL]+REQUIRED_FEATURES)
    if err: return None, dbc.Alert(err, color="danger", dismissable=True)
    return df.to_json(date_format="iso", orient="split"), dbc.Alert(f"Loaded: {filename} • Rows: {len(df):,}", color="success")

@app.callback(
    Output("store_real_df","data"), Output("real_info","children"),
    Input("upload_real","contents"), State("upload_real","filename"), prevent_initial_call=True)
def handle_upload_real(contents, filename):
    if not contents: return no_update, no_update
    df, err = parse_contents(contents, filename, required_cols=None)
    if err: return None, dbc.Alert(err, color="danger", dismissable=True)
    return df.to_json(date_format="iso", orient="split"), dbc.Alert(f"Loaded: {filename} • Rows: {len(df):,}", color="success")

@app.callback(
    Output("preview_tables","children"),
    Input("store_sim_df","data"), Input("store_real_df","data"))
def preview_tables(sim_json, real_json):
    children=[]
    if sim_json:
        df=pd.read_json(sim_json, orient="split")
        children += [html.H6("Simulation Preview"),
                     dash_table.DataTable(columns=[{"name":c,"id":c} for c in df.columns],
                                          data=df.head(10).to_dict("records"),
                                          style_table={"overflowX":"auto"}, page_size=10)]
    if real_json:
        df=pd.read_json(real_json, orient="split")
        children += [html.H6("Real/Sensor Preview"),
                     dash_table.DataTable(columns=[{"name":c,"id":c} for c in df.columns],
                                          data=df.head(10).to_dict("records"),
                                          style_table={"overflowX":"auto"}, page_size=10)]
    return children or html.Div(className="text-muted", children="No data uploaded yet.")

@app.callback(
    Output("result_table","data"), Output("stacked_bar","figure"),
    Output("reduction_bar","figure"), Output("timeseries","figure"),
    Output("store_result_df","data"),
    Input("store_sim_df","data"), Input("store_real_df","data"),
    Input("ef","value"), Input("share_light","value"), Input("share_hvac","value"),
    Input("sav_light","value"), Input("sav_hvac","value"),
    Input("pv_offset","value"), Input("embodied","value"))
def update_all(sim_json, real_json, ef, sL, sH, svL, svH, pv, emb):
    sim_df = pd.read_json(sim_json, orient="split") if sim_json else None
    real_df= pd.read_json(real_json, orient="split") if real_json else None
    if sim_df is not None and TARGET_COL in sim_df.columns:
        E_base=float(sim_df[TARGET_COL].sum())
    elif real_df is not None and TARGET_COL in real_df.columns:
        E_base=float(real_df[TARGET_COL].sum())
    else:
        empty=pd.DataFrame(columns=["Scenario","Embodied_CO2 (t)","Operational_CO2 (t)","Total_CO2 (t)",
                                    "Reduction_vs_Base (%)","Energy (kWh)"])
        msg=go.Figure().update_layout(title="Provide dataset with ‘Energy Consumption (kWh)’")
        return empty.to_dict("records"), msg, msg, make_time_series(None,None), empty.to_json(orient="split")
    scen=scenario_energies(E_base, sL, sH, svL, svH, pv)
    df_res=make_result_table(scen, ef, emb)
    return df_res.to_dict("records"), make_horizontal_bar(df_res), make_reduction_bar(df_res), \
           make_time_series(sim_df, real_df), df_res.to_json(date_format="iso", orient="split")

# =========================
# XGBoost – Sim vs Real
# =========================
@app.callback(
    Output("ml_compare","figure"), Output("ml_metrics","data"),
    Input("store_sim_df","data"), Input("store_real_df","data"), prevent_initial_call=True)
def xgb_sim_vs_real(sim_json, real_json):
    if not sim_json or not real_json:
        return go.Figure().update_layout(title="Upload both Simulation & Real datasets."), []
    sim = pd.read_json(sim_json, orient="split")
    real= pd.read_json(real_json, orient="split")
    miss_sim=[c for c in [TARGET_COL]+REQUIRED_FEATURES if c not in sim.columns]
    miss_real=[c for c in REQUIRED_FEATURES if c not in real.columns]
    if miss_sim or miss_real:
        msg=[]
        if miss_sim: msg.append(f"Simulation missing: {miss_sim}")
        if miss_real: msg.append(f"Real missing: {miss_real}")
        return go.Figure().update_layout(title="Column mismatch — "+" | ".join(msg)), []

    X_sim=sim[REQUIRED_FEATURES]; y_sim=sim[TARGET_COL].values
    X_real=real[REQUIRED_FEATURES]
    model=XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=6,
                       subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    model.fit(X_sim, y_sim)
    y_pred=model.predict(X_real)
    m=min(len(y_pred), len(y_sim))
    y_pred=y_pred[:m]; y_sim=y_sim[:m]; hours=np.arange(1,m+1)
    mae=float(mean_absolute_error(y_sim, y_pred))
    rmse=float(np.sqrt(mean_squared_error(y_sim, y_pred)))
    fig=go.Figure()
    fig.add_scatter(x=hours, y=y_sim, mode="lines+markers", name="Simulation (DesignBuilder)",
                    line=dict(dash="dash"), marker=dict(symbol="x"))
    fig.add_scatter(x=hours, y=y_pred, mode="lines+markers", name="Predicted from Real (XGBoost)")
    fig.update_layout(title="Simulation vs Predicted (from Real Sensors) — XGBoost",
                      xaxis_title="Hour Index", yaxis_title="Energy (kWh)",
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                      annotations=[dict(xref="paper", yref="paper", x=0.01, y=0.98,
                                        text=f"MAE: {mae:.2f}<br>RMSE: {rmse:.2f}",
                                        showarrow=False, bgcolor="white", bordercolor="gray")],
                      height=460)
    metrics=[{"metric":"MAE (kWh)","value":f"{mae:.2f}"},{"metric":"RMSE (kWh)","value":f"{rmse:.2f}"},
             {"metric":"n (hours)","value":f"{m}"}]
    return fig, metrics

# =========================
# K-Fold CV – button-triggered (Fast/Full)
# =========================
def build_transformer(n_features):
    inp = layers.Input(shape=(1, n_features))
    x = layers.LayerNormalization(epsilon=1e-6)(inp)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.2)(x, x)
    x = layers.Dropout(0.2)(x); x = x + inp
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Conv1D(128, 1, activation="relu")(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Conv1D(inp.shape[-1], 1)(y)
    x = x + y
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

def build_gru(n_features):
    inp = layers.Input(shape=(1, n_features))
    x = layers.GRU(64)(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out)
    m.compile(optimizer="adam", loss="mse")
    return m

@app.callback(
    Output("cv_status","children"),
    Output("cv_bar_metrics","figure"),
    Output("cv_overview_table","data"),
    Input("run_cv_btn","n_clicks"),
    State("store_sim_df","data"),
    State("cv_mode","value"), State("cv_folds","value"),
    State("cv_limit","value"), State("cv_dl","value"),
    prevent_initial_call=True)
def run_cv(n_clicks, sim_json, mode, folds, limit, dl_flags):
    if not sim_json:
        return "Upload Simulation first.", go.Figure(), []
    status="Starting …"; fig=go.Figure(); table=[]
    sim=pd.read_json(sim_json, orient="split")
    if TARGET_COL not in sim.columns:
        return "Simulation missing target column.", fig, table

    # Sample limit for speed
    df = sim.copy()
    if isinstance(limit, (int, float)) and limit and limit>0 and len(df)>limit:
        df = df.iloc[:int(limit)].copy()

    # lag features
    df["Energy_Lag1"]=df[TARGET_COL].shift(1)
    df["Energy_Lag2"]=df[TARGET_COL].shift(2)
    df=df.dropna().reset_index(drop=True)

    # feature set (all numeric except target & time-ish cols)
    drop_cols=[TARGET_COL]
    for c in ["Hour","hour","Time","Datetime","Timestamp"]:
        if c in df.columns: drop_cols.append(c)
    num=df.select_dtypes(include="number")
    X=num.drop(columns=[c for c in drop_cols if c in num.columns], errors="ignore").values
    y=df[TARGET_COL].values.reshape(-1,1)

    if X.shape[1]==0: return "No usable numeric features.", fig, table

    # CV setup
    folds = 3 if mode=="fast" else max(5, int(folds or 5))
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    models_to_run=["XGBoost","SVR"]
    include_dl = ("dl" in (dl_flags or [])) and TF_AVAILABLE
    if include_dl: models_to_run += ["GRU","Transformer"]

    results=[]  # rows for table
    metric_plot = {"model":[], "MAE":[], "RMSE":[], "R2":[]}

    # helpers
    def mean_std(a): return float(np.mean(a)), float(np.std(a, ddof=1))

    for model_name in models_to_run:
        status=f"Running {model_name} ({folds}-fold)…"
        mae_list=[]; rmse_list=[]; r2_list=[]
        for tr_idx, te_idx in kf.split(X):
            Xtr_raw, Xte_raw = X[tr_idx], X[te_idx]
            ytr_raw, yte_raw = y[tr_idx], y[te_idx]
            sx, sy = MinMaxScaler(), MinMaxScaler()
            Xtr=sx.fit_transform(Xtr_raw); Xte=sx.transform(Xte_raw)
            ytr=sy.fit_transform(ytr_raw); yte=sy.transform(yte_raw)

            if model_name=="XGBoost":
                est = XGBRegressor(n_estimators=300 if mode=="fast" else 600,
                                   learning_rate=0.08, max_depth=6,
                                   subsample=0.9, colsample_bytree=0.9,
                                   random_state=42, n_jobs=-1)
                est.fit(Xtr, ytr.ravel())
                pred = sy.inverse_transform(est.predict(Xte).reshape(-1,1)).ravel()

            elif model_name=="SVR":
                est = SKSVR(kernel='rbf', C=40.0, epsilon=0.1, gamma='scale')
                est.fit(Xtr, ytr.ravel())
                pred = sy.inverse_transform(est.predict(Xte).reshape(-1,1)).ravel()

            elif model_name=="GRU":
                epochs = 4 if mode=="fast" else 12
                Xtr3=np.expand_dims(Xtr,1); Xte3=np.expand_dims(Xte,1)
                m=build_gru(Xtr.shape[1])
                es=callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
                m.fit(Xtr3, ytr, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
                pred = sy.inverse_transform(m.predict(Xte3, verbose=0)).ravel()

            elif model_name=="Transformer":
                epochs = 4 if mode=="fast" else 12
                Xtr3=np.expand_dims(Xtr,1); Xte3=np.expand_dims(Xte,1)
                m=build_transformer(Xtr.shape[1])
                es=callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
                m.fit(Xtr3, ytr, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
                pred = sy.inverse_transform(m.predict(Xte3, verbose=0)).ravel()

            else:
                continue

            true = yte_raw.ravel()
            mae_list.append(mean_absolute_error(true, pred))
            rmse_list.append(np.sqrt(mean_squared_error(true, pred)))
            r2_list.append(r2_score(true, pred))

        mae_m, mae_s = mean_std(mae_list)
        rmse_m, rmse_s = mean_std(rmse_list)
        r2_m, r2_s = mean_std(r2_list)

        results.append({"model":model_name, "mae":round(mae_m,2), "rmse":round(rmse_m,2), "r2":round(r2_m,4)})
        metric_plot["model"].append(model_name)
        metric_plot["MAE"].append(mae_m); metric_plot["RMSE"].append(rmse_m); metric_plot["R2"].append(r2_m)

    # build bar figure: three small bars per model (MAE/RMSE/R2 scaled)
    dfm = pd.DataFrame(metric_plot)
    fig = go.Figure()
    fig.add_bar(x=dfm["model"], y=dfm["MAE"], name="MAE (kWh)", marker_color="#4e79a7")
    fig.add_bar(x=dfm["model"], y=dfm["RMSE"], name="RMSE (kWh)", marker_color="#f28e2b")
    # R2 on secondary axis
    fig.update_layout(title=f"K-Fold Metrics (mode: {mode}, folds: {folds}, rows: {len(df):,})",
                      barmode="group", xaxis_title="Model", yaxis_title="Error (kWh)",
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), height=420)
    # add R2 line
    fig.add_trace(go.Scatter(x=dfm["model"], y=dfm["R2"], mode="lines+markers",
                             name="R²", marker=dict(symbol="diamond", size=9), yaxis="y2"))
    fig.update_layout(yaxis2=dict(title="R²", overlaying="y", side="right", range=[0,1]))

    status="Done."
    # sort by MAE
    table = sorted(results, key=lambda r: r["mae"])
    return status, fig, table

# =========================
# Save/Download
# =========================
@app.callback(
    Output("save_msg","children"), Output("dl_csv","data"), Output("dl_json","data"),
    Input("save_btn","n_clicks"), Input("dl_csv_btn","n_clicks"), Input("dl_json_btn","n_clicks"),
    State("store_result_df","data"),
    State("ef","value"), State("share_light","value"), State("share_hvac","value"),
    State("sav_light","value"), State("sav_hvac","value"),
    State("pv_offset","value"), State("embodied","value"),
    prevent_initial_call=True)
def save_and_download(n_save, n_csv, n_json, result_json, ef, sL, sH, svL, svH, pv, emb):
    if not result_json: return no_update, None, None
    btn=ctx.triggered_id; df=pd.read_json(result_json, orient="split")
    now=datetime.now().strftime("%Y%m%d_%H%M%S")
    meta={"timestamp":now, "parameters":{"EF (kg/kWh)":ef,"Share Light":sL,"Share HVAC":sH,
          "Saving Light":svL,"Saving HVAC":svH,"PV offset":pv,"Embodied (t)":emb}}
    if btn=="save_btn":
        out=ensure_dir("./exports")
        df.to_csv(os.path.join(out, f"scenario_results_{now}.csv"), index=False)
        with open(os.path.join(out, f"scenario_results_{now}.json"), "w", encoding="utf-8") as f:
            json.dump({"meta":meta, "results":df.to_dict(orient="records")}, f, ensure_ascii=False, indent=2)
        return "Results saved to ./exports (CSV & JSON).", None, None
    if btn=="dl_csv_btn":
        return no_update, dcc.send_data_frame(df.to_csv, f"scenario_results_{now}.csv", index=False), None
    if btn=="dl_json_btn":
        payload={"meta":meta, "results":df.to_dict(orient="records")}
        return no_update, None, dict(content=json.dumps(payload, ensure_ascii=False, indent=2),
                                     filename=f"scenario_results_{now}.json")
    return no_update, None, None

# =========================
# Main
# =========================
import os
import dash
from dash import Dash, dcc, html

# Dash app initialize
app = Dash(__name__)
server = app.server   # Gunicorn এইটা ধরবে

