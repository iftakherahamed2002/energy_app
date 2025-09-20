# -*- coding: utf-8 -*-
"""
Construction Energy & CO₂ Dashboard – KUET Textile Thesis
Author: Iftakher Ahamed

Professional single-file Plotly Dash app for stakeholder scenario analysis.
- Upload Simulation (DesignBuilder) and optional Real/Sensor datasets
- Adjust assumptions: grid emission factor, load shares, savings, PV offset, embodied CO₂
- Visuals: Horizontal grouped bar (Embodied / Operational / Total), % reduction bar, time series
- ML tab (XGBoost): compares Simulation vs Predicted (from Real features)
- ML tab (Transformer): trains on Simulation & shows Actual vs Predicted with MAE/RMSE/R² box
- Export: CSV / JSON, plus local persistence
"""

import os
import io
import json
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

# --- ML (XGBoost) ---
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # R² needed for Transformer tab
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- ML (Keras Transformer) ---
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------
# App & Server
# ---------------------------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    title="Construction Energy & CO₂ – KUET Textile Thesis"
)
server = app.server

# ---------------------------
# Constants / Utility
# ---------------------------
REQUIRED_FEATURES = [
    "Wind Speed (m/s)",
    "Relative Humidity (%)",
    "Outdoor Temperature (°C)",
    "Indoor Temperature (°C)",
    "Lighting (LUX)",
    "Solar Radiation (kw/m²)",
]
TARGET_COL = "Energy Consumption (kWh)"

def ensure_dir(path="./exports"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def parse_contents(contents, filename, required_cols=None):
    """Decode the uploaded file to a DataFrame with light validation."""
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

    # normalize headers (double-spaces etc.)
    df.columns = [c.replace("  ", " ").strip() for c in df.columns]

    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Missing required columns in {filename}: {missing}"

    return df, None

def scenario_energies(E_base, share_light, share_hvac, sav_light, sav_hvac, pv_offset):
    """Compute scenario energy (kWh) from a baseline."""
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

def kwh_to_tonnes(kwh, ef_kg_per_kwh):
    return (kwh * ef_kg_per_kwh) / 1000.0

def make_result_table(scen_kwh, ef, embodied_tonnes):
    labels = list(scen_kwh.keys())
    operational_t = {k: kwh_to_tonnes(v, ef) for k, v in scen_kwh.items()}
    total_t       = {k: embodied_tonnes + operational_t[k] for k in labels}
    base_total    = total_t["Base Case"]
    reduction_pct = {k: (1 - total_t[k] / base_total) * 100 for k in labels}

    df = pd.DataFrame({
        "Scenario": labels,
        "Embodied_CO2 (t)": [embodied_tonnes] * len(labels),
        "Operational_CO2 (t)": [operational_t[k] for k in labels],
        "Total_CO2 (t)": [total_t[k] for k in labels],
        "Reduction_vs_Base (%)": [reduction_pct[k] for k in labels],
        "Energy (kWh)": [scen_kwh[k] for k in labels],
    })
    return df

# ---------- Horizontal grouped bar ----------
def make_horizontal_bar(df):
    scenarios   = df["Scenario"]
    embodied    = df["Embodied_CO2 (t)"]
    operational = df["Operational_CO2 (t)"]
    total       = df["Total_CO2 (t)"]

    fig = go.Figure()
    fig.add_bar(
        y=scenarios, x=embodied, orientation='h',
        name="Embodied", marker_color="#1f77b4",
        text=[f"{v:.1f}" for v in embodied], textposition="outside",
        hovertemplate="Scenario: %{y}<br>Embodied: %{x:.2f} t<extra></extra>",
    )
    fig.add_bar(
        y=scenarios, x=operational, orientation='h',
        name="Operational", marker_color="#ff7f0e",
        text=[f"{v:.1f}" for v in operational], textposition="outside",
        hovertemplate="Scenario: %{y}<br>Operational: %{x:.2f} t<extra></extra>",
    )
    fig.add_bar(
        y=scenarios, x=total, orientation='h',
        name="Total", marker_color="#2ca02c",
        text=[f"{v:.1f}" for v in total], textposition="outside",
        hovertemplate="Scenario: %{y}<br>Total: %{x:.2f} t<extra></extra>",
    )

    fig.update_layout(
        barmode="group",
        title="Embodied vs Operational vs Total CO₂",
        xaxis_title="CO₂ Emissions (tonnes)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(l=110, r=40, t=60, b=90),
        height=520
    )
    max_total = float(np.nanmax(total)) if len(total) else 0
    fig.update_xaxes(range=[0, max_total * 1.20 if max_total > 0 else 10])
    return fig

def make_reduction_bar(df):
    fig = px.bar(
        df, x="Scenario", y="Reduction_vs_Base (%)",
        title="Reduction vs Base (%) – Total CO₂",
    )
    fig.update_layout(
        yaxis_title="Reduction (%)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )
    fig.update_traces(hovertemplate="Scenario: %{x}<br>Reduction: %{y:.2f}%<extra></extra>")
    return fig

def make_time_series(sim_df=None, real_df=None):
    fig = go.Figure()
    added = False

    if sim_df is not None and TARGET_COL in sim_df.columns:
        y_sim = sim_df[TARGET_COL].values
        x_sim = np.arange(1, len(y_sim) + 1)
        fig.add_scatter(x=x_sim, y=y_sim, mode="lines", name="Simulation Energy (kWh)",
                        line=dict(dash="dash"),
                        hovertemplate="Index: %{x}<br>Simulation: %{y:.2f} kWh<extra></extra>")
        added = True

    if real_df is not None and TARGET_COL in real_df.columns:
        y_real = real_df[TARGET_COL].values
        x_real = np.arange(1, len(y_real) + 1)
        fig.add_scatter(x=x_real, y=y_real, mode="lines+markers", name="Real/Sensor Energy (kWh)",
                        hovertemplate="Index: %{x}<br>Real: %{y:.2f} kWh<extra></extra>")
        added = True

    if not added:
        fig.update_layout(
            title="Upload a dataset containing ‘Energy Consumption (kWh)’ to view the time series.",
            margin=dict(l=40, r=20, t=60, b=40)
        )
        return fig

    fig.update_layout(
        title="Energy Time Series (Simulation vs Real)",
        xaxis_title="Hour Index",
        yaxis_title="Energy (kWh)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )
    return fig

# ---------------------------
# UI Components
# ---------------------------
banner = dbc.Alert(
    [
        html.H3("Construction Energy & CO₂ Dashboard — KUET Textile Thesis", className="mb-1"),
        html.Div("Interactive scenario analysis for embodied and operational emissions.", className="mb-0"),
    ],
    color="info", className="mb-4"
)

upload_card = dbc.Card(
    dbc.CardBody([
        html.H5("Upload Data", className="card-title"),
        html.P(
            "Upload Simulation (DesignBuilder) and optional Real/Sensor datasets. "
            "Required columns (case-sensitive):",
            className="mb-1"
        ),
        html.Ul([
            html.Li(html.Code(TARGET_COL)),
            html.Li(", ".join(REQUIRED_FEATURES))
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Simulation Excel/CSV"),
                dcc.Upload(
                    id="upload_sim",
                    children=html.Div(['Drag & Drop or ', html.A('Select Simulation File')]),
                    multiple=False,
                    style={
                        "width": "100%", "height": "70px", "lineHeight": "70px",
                        "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "6px",
                        "textAlign": "center"
                    }
                ),
                html.Div(id="sim_info", className="mt-2")
            ], md=6),
            dbc.Col([
                dbc.Label("Real/Sensor Excel/CSV (optional)"),
                dcc.Upload(
                    id="upload_real",
                    children=html.Div(['Drag & Drop or ', html.A('Select Real File')]),
                    multiple=False,
                    style={
                        "width": "100%", "height": "70px", "lineHeight": "70px",
                        "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "6px",
                        "textAlign": "center"
                    }
                ),
                html.Div(id="real_info", className="mt-2")
            ], md=6),
        ], className="g-3"),
        html.Hr(className="my-3"),
        html.Div(id="preview_tables")
    ])
)

controls_card = dbc.Card(
    dbc.CardBody([
        html.H5("Parameters", className="card-title"),
        html.Div("Set assumptions (adjustable by stakeholders):", className="text-muted mb-2"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Grid Emission Factor (kg CO₂/kWh)"),
                dcc.Slider(id="ef", min=0.30, max=1.20, step=0.01, value=0.70,
                           tooltip={"placement": "bottom"})
            ], width=12)
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Lighting share of load"),
                dcc.Slider(id="share_light", min=0.00, max=0.80, step=0.01, value=0.25)
            ], md=6),
            dbc.Col([
                dbc.Label("HVAC share of load"),
                dcc.Slider(id="share_hvac", min=0.00, max=0.90, step=0.01, value=0.50)
            ], md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Lighting saving (%)"),
                dcc.Slider(id="sav_light", min=0.00, max=0.50, step=0.01, value=0.20)
            ], md=6),
            dbc.Col([
                dbc.Label("HVAC saving (%)"),
                dcc.Slider(id="sav_hvac", min=0.00, max=0.50, step=0.01, value=0.15)
            ], md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("PV offset (% of total)"),
                dcc.Slider(id="pv_offset", min=0.00, max=0.50, step=0.01, value=0.15)
            ], md=6),
            dbc.Col([
                dbc.Label("Embodied CO₂ (tonnes)"),
                dcc.Slider(id="embodied", min=0, max=500, step=0.1, value=137.7)
            ], md=6),
        ], className="mb-2"),
        html.Small(
            "Shares represent contribution to total consumption. Savings are applied to those shares.",
            className="text-muted"
        ),
        html.Hr(),
        dbc.Button("Save Results", id="save_btn", color="primary", className="me-2"),
        dbc.Button("Export CSV", id="dl_csv_btn", color="secondary", className="me-2"),
        dbc.Button("Export JSON", id="dl_json_btn", color="secondary"),
        dcc.Download(id="dl_csv"),
        dcc.Download(id="dl_json"),
        html.Div(id="save_msg", className="mt-2 text-success")
    ])
)

graphs_card = dbc.Card(
    dbc.CardBody([
        html.H5("Visualizations", className="card-title"),
        dcc.Tabs([
            dcc.Tab(label="CO₂ Scenarios (Embodied, Operational, Total)", children=[
                dcc.Graph(id="stacked_bar"),
                dcc.Graph(id="reduction_bar")
            ]),
            dcc.Tab(label="Energy Time Series", children=[
                dcc.Graph(id="timeseries")
            ]),
            dcc.Tab(label="ML Comparison (XGBoost)", children=[
                html.Div(className="mb-2", children=html.Small(
                    "Trains XGBoost on Simulation features and predicts energy from Real/Sensor features. "
                    "Runs automatically when both files are uploaded."
                )),
                dcc.Graph(id="ml_compare"),
                dash_table.DataTable(
                    id="ml_metrics",
                    columns=[{"name": "METRIC", "id": "metric"},
                             {"name": "VALUE",  "id": "value"}],
                    data=[],
                    style_table={"overflowX": "auto"},
                    style_cell={"minWidth": 120},
                    page_size=5
                )
            ]),
            dcc.Tab(label="ML (Transformer)", children=[
                html.Div(className="mb-2", children=html.Small(
                    "Trains a Transformer on the Simulation dataset (uses lag features) and shows Actual vs Predicted with MAE/RMSE/R²."
                )),
                dcc.Graph(id="tf_compare"),
                dash_table.DataTable(
                    id="tf_metrics",
                    columns=[{"name": "METRIC", "id": "metric"},
                             {"name": "VALUE",  "id": "value"}],
                    data=[],
                    style_table={"overflowX": "auto"},
                    style_cell={"minWidth": 120},
                    page_size=5
                )
            ]),
        ])
    ])
)

table_card = dbc.Card(
    dbc.CardBody([
        html.H5("Scenario Results Table", className="card-title"),
        dash_table.DataTable(
            id="result_table",
            columns=[
                {"name": "Scenario",              "id": "Scenario"},
                {"name": "Embodied_CO2 (t)",      "id": "Embodied_CO2 (t)",      "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Operational_CO2 (t)",   "id": "Operational_CO2 (t)",   "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Total_CO2 (t)",         "id": "Total_CO2 (t)",         "type": "numeric", "format": {"specifier": ".2f"}},
                {"name": "Reduction_vs_Base (%)", "id": "Reduction_vs_Base (%)", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Energy (kWh)",          "id": "Energy (kWh)",          "type": "numeric", "format": {"specifier": ".0f"}},
            ],
            data=[],
            style_table={"overflowX": "auto"},
            style_cell={"minWidth": 140, "whiteSpace": "normal"},
            page_size=10
        )
    ])
)

stores = html.Div([
    dcc.Store(id="store_sim_df"),
    dcc.Store(id="store_real_df"),
    dcc.Store(id="store_result_df"),
])

footer = html.Div(
    className="text-muted mt-3",
    children="Developer : Iftakher Ahamed"
)

app.layout = dbc.Container([
    banner,
    stores,
    dbc.Row([
        dbc.Col(upload_card, lg=7),
        dbc.Col(controls_card, lg=5),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(graphs_card, lg=7),
        dbc.Col(table_card, lg=5),
    ]),
    footer
], fluid=True)

# ---------------------------
# Callbacks
# ---------------------------
@app.callback(
    Output("store_sim_df", "data"),
    Output("sim_info", "children"),
    Input("upload_sim", "contents"),
    State("upload_sim", "filename"),
    prevent_initial_call=True
)
def handle_upload_sim(contents, filename):
    if contents is None:
        return no_update, no_update
    df, err = parse_contents(contents, filename, required_cols=[TARGET_COL] + REQUIRED_FEATURES)
    if err:
        return None, dbc.Alert(err, color="danger", dismissable=True)
    return df.to_json(date_format="iso", orient="split"), dbc.Alert(
        f"Loaded: {filename}  •  Rows: {len(df):,}", color="success"
    )

@app.callback(
    Output("store_real_df", "data"),
    Output("real_info", "children"),
    Input("upload_real", "contents"),
    State("upload_real", "filename"),
    prevent_initial_call=True
)
def handle_upload_real(contents, filename):
    if contents is None:
        return no_update, no_update
    df, err = parse_contents(contents, filename, required_cols=None)
    if err:
        return None, dbc.Alert(err, color="danger", dismissable=True)
    return df.to_json(date_format="iso", orient="split"), dbc.Alert(
        f"Loaded: {filename}  •  Rows: {len(df):,}", color="success"
    )

@app.callback(
    Output("preview_tables", "children"),
    Input("store_sim_df", "data"),
    Input("store_real_df", "data")
)
def preview_tables(sim_json, real_json):
    children = []
    if sim_json:
        df_sim = pd.read_json(sim_json, orient="split")
        children += [
            html.H6("Simulation Preview"),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df_sim.columns],
                data=df_sim.head(10).to_dict("records"),
                style_table={"overflowX": "auto"},
                page_size=10
            )
        ]
    if real_json:
        df_real = pd.read_json(real_json, orient="split")
        children += [
            html.H6("Real/Sensor Preview"),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df_real.columns],
                data=df_real.head(10).to_dict("records"),
                style_table={"overflowX": "auto"},
                page_size=10
            )
        ]
    if not children:
        return html.Div(className="text-muted", children="No data uploaded yet.")
    return children

@app.callback(
    Output("result_table", "data"),
    Output("stacked_bar", "figure"),
    Output("reduction_bar", "figure"),
    Output("timeseries", "figure"),
    Output("store_result_df", "data"),
    Input("store_sim_df", "data"),
    Input("store_real_df", "data"),
    Input("ef", "value"),
    Input("share_light", "value"),
    Input("share_hvac", "value"),
    Input("sav_light", "value"),
    Input("sav_hvac", "value"),
    Input("pv_offset", "value"),
    Input("embodied", "value"),
)
def update_all(sim_json, real_json, ef, share_light, share_hvac, sav_light, sav_hvac, pv_offset, embodied):
    sim_df  = pd.read_json(sim_json, orient="split") if sim_json else None
    real_df = pd.read_json(real_json, orient="split") if real_json else None

    if sim_df is not None and TARGET_COL in sim_df.columns:
        E_base = float(sim_df[TARGET_COL].sum())
    elif real_df is not None and TARGET_COL in real_df.columns:
        E_base = float(real_df[TARGET_COL].sum())
    else:
        empty_df = pd.DataFrame(columns=[
            "Scenario", "Embodied_CO2 (t)", "Operational_CO2 (t)", "Total_CO2 (t)",
            "Reduction_vs_Base (%)", "Energy (kWh)"
        ])
        msg_fig = go.Figure().update_layout(
            title="Provide a dataset with ‘Energy Consumption (kWh)’ to view CO₂ charts."
        )
        ts = make_time_series(None, None)
        return empty_df.to_dict("records"), msg_fig, msg_fig, ts, empty_df.to_json(orient="split")

    scen_kwh = scenario_energies(
        E_base,
        share_light=share_light, share_hvac=share_hvac,
        sav_light=sav_light,   sav_hvac=sav_hvac,
        pv_offset=pv_offset
    )

    df_res = make_result_table(scen_kwh, ef=ef, embodied_tonnes=embodied)

    fig_hbar = make_horizontal_bar(df_res)
    fig_red  = make_reduction_bar(df_res)
    ts       = make_time_series(sim_df, real_df)

    return (
        df_res.to_dict("records"),
        fig_hbar,
        fig_red,
        ts,
        df_res.to_json(date_format="iso", orient="split"),
    )

# --------- ML Comparison (XGBoost) ---------
@app.callback(
    Output("ml_compare", "figure"),
    Output("ml_metrics", "data"),
    Input("store_sim_df", "data"),
    Input("store_real_df", "data"),
    prevent_initial_call=True
)
def xgb_sim_vs_real(sim_json, real_json):
    if not sim_json or not real_json:
        fig = go.Figure().update_layout(
            title="Upload both Simulation and Real/Sensor datasets to run XGBoost comparison."
        )
        return fig, []

    sim_df  = pd.read_json(sim_json, orient="split")
    real_df = pd.read_json(real_json, orient="split")

    # Column check
    missing_sim = [c for c in [TARGET_COL] + REQUIRED_FEATURES if c not in sim_df.columns]
    missing_real_features = [c for c in REQUIRED_FEATURES if c not in real_df.columns]
    if missing_sim or missing_real_features:
        msg = []
        if missing_sim:
            msg.append(f"Simulation missing: {missing_sim}")
        if missing_real_features:
            msg.append(f"Real missing: {missing_real_features}")
        fig = go.Figure().update_layout(title="Column mismatch — " + " | ".join(msg))
        return fig, []

    # Split features/target
    X_sim = sim_df[REQUIRED_FEATURES]
    y_sim = sim_df[TARGET_COL].values
    X_real = real_df[REQUIRED_FEATURES]

    # Small grid to stay fast
    xgb = XGBRegressor(random_state=42)
    param_grid = {
        "n_estimators": [200, 400],
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 2]
    }
    gs = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        verbose=0
    )
    gs.fit(X_sim, y_sim)
    best = gs.best_estimator_

    # Predict with real features
    y_real_pred = best.predict(X_real)

    # Align lengths
    m = min(len(y_real_pred), len(y_sim))
    y_real_pred = y_real_pred[:m]
    y_sim_slice = y_sim[:m]
    hours = np.arange(1, m+1)

    # Metrics (R² not shown here as per earlier choice)
    mae = float(mean_absolute_error(y_sim_slice, y_real_pred))
    rmse = float(np.sqrt(mean_squared_error(y_sim_slice, y_real_pred)))
    metrics_rows = [
        {"metric": "MAE (kWh)",  "value": f"{mae:.2f}"},
        {"metric": "RMSE (kWh)", "value": f"{rmse:.2f}"},
        {"metric": "n (hours)",  "value": f"{m}"}
    ]

    # Plot
    fig = go.Figure()
    fig.add_scatter(
        x=hours, y=y_sim_slice, mode="lines+markers", name="Simulation Energy (DesignBuilder)",
        line=dict(dash="dash"), marker=dict(symbol="x"),
        hovertemplate="Hour: %{x}<br>Simulation: %{y:.2f} kWh<extra></extra>"
    )
    fig.add_scatter(
        x=hours, y=y_real_pred, mode="lines+markers", name="Predicted from Real Sensors (XGBoost)",
        hovertemplate="Hour: %{x}<br>Predicted: %{y:.2f} kWh<extra></extra>"
    )
    fig.update_layout(
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.01, y=0.98,
                text=f"MAE: {mae:.2f}<br>RMSE: {rmse:.2f}",
                showarrow=False, align="left",
                bgcolor="white", bordercolor="gray", borderwidth=1,
                opacity=0.9, font=dict(size=12, color="#111")
            )
        ],
        title="Simulation vs Predicted (from Real Sensors) — XGBoost",
        xaxis_title="Hour Index",
        yaxis_title="Energy Consumption (kWh)",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=60, b=80),
        height=460
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig, metrics_rows

# --------- ML (Transformer trained on Simulation) ---------
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs  # skip
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

@app.callback(
    Output("tf_compare", "figure"),
    Output("tf_metrics", "data"),
    Input("store_sim_df", "data"),
    prevent_initial_call=True
)
def run_transformer(sim_json):
    if not sim_json:
        fig = go.Figure().update_layout(
            title="Upload a Simulation dataset to run the Transformer model."
        )
        return fig, []

    sim_df = pd.read_json(sim_json, orient="split")

    # Must have target; features optional (we'll use whatever except Energy & Hour)
    if TARGET_COL not in sim_df.columns:
        fig = go.Figure().update_layout(
            title=f"Missing column: {TARGET_COL}"
        )
        return fig, []

    # Build lag features
    df = sim_df.copy()
    df["Energy_Lag1"] = df[TARGET_COL].shift(1)
    df["Energy_Lag2"] = df[TARGET_COL].shift(2)
    df = df.dropna()

    # Features: use all numeric columns except target and (optional) Hour-like column
    drop_cols = [TARGET_COL]
    # Drop common time index columns if present
    for col in ["Hour", "hour", "Time", "Datetime", "Timestamp"]:
        if col in df.columns:
            drop_cols.append(col)

    # Keep only numeric to be safe
    num_df = df.select_dtypes(include=["number"]).copy()
    X = num_df.drop(columns=[c for c in drop_cols if c in num_df.columns], errors="ignore")
    y = df[TARGET_COL].values.reshape(-1, 1)

    if X.shape[1] == 0:
        fig = go.Figure().update_layout(
            title="No usable numeric features found after removing target/time columns."
        )
        return fig, []

    # Scale
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Reshape to (samples, timesteps=1, features)
    X_train = np.expand_dims(X_train, axis=1)
    X_test  = np.expand_dims(X_test, axis=1)

    # Model
    inputs = layers.Input(shape=X_train.shape[1:])
    x = transformer_encoder(inputs, head_size=64, num_heads=6, ff_dim=128, dropout=0.3)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3, factor=0.1, min_lr=1e-4
    )

    # Train (keep it moderate)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=60, batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    # Predict & inverse scale
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = scaler_y.inverse_transform(y_test).reshape(-1)

    # Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Plot (first 200 samples)
    nshow = min(200, len(y_true))
    xs = np.arange(1, nshow + 1)

    fig = go.Figure()
    fig.add_scatter(
        x=xs, y=y_true[:nshow],
        mode="lines", name="Actual",
        line=dict(width=2, color="black"),
        hovertemplate="Index: %{x}<br>Actual: %{y:.2f} kWh<extra></extra>"
    )
    fig.add_scatter(
        x=xs, y=y_pred[:nshow],
        mode="lines", name="Transformer Prediction",
        line=dict(dash="dash"),
        hovertemplate="Index: %{x}<br>Predicted: %{y:.2f} kWh<extra></extra>"
    )

    # ---- metrics info-box (matches your Step 10 intent) ----
    fig.update_layout(
        annotations=[
            dict(
                xref="paper", yref="paper",
                x=0.01, y=0.98,
                text=f"MAE: {mae:.4f}<br>RMSE: {rmse:.4f}<br>R²: {r2:.2f}",
                showarrow=False, align="left",
                bgcolor="white", bordercolor="gray", borderwidth=1,
                opacity=0.9, font=dict(size=12, color="#111")
            )
        ],
        title="Transformer: Prediction vs Actual (First 200 Samples)",
        xaxis_title="Hour Index (sample)",
        yaxis_title="Energy Consumption (kWh)",
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=60, b=80),
        height=460
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    metrics_rows = [
        {"metric": "MAE (kWh)",  "value": f"{mae:.4f}"},
        {"metric": "RMSE (kWh)", "value": f"{rmse:.4f}"},
        {"metric": "R²",         "value": f"{r2:.2f}"},
        {"metric": "n (test)",   "value": f"{len(y_true)}"},
    ]
    return fig, metrics_rows
# -------------------------------------------

@app.callback(
    Output("save_msg", "children"),
    Output("dl_csv", "data"),
    Output("dl_json", "data"),
    Input("save_btn", "n_clicks"),
    Input("dl_csv_btn", "n_clicks"),
    Input("dl_json_btn", "n_clicks"),
    State("store_result_df", "data"),
    State("ef", "value"),
    State("share_light", "value"),
    State("share_hvac", "value"),
    State("sav_light", "value"),
    State("sav_hvac", "value"),
    State("pv_offset", "value"),
    State("embodied", "value"),
    prevent_initial_call=True
)
def save_and_download(n_save, n_csv, n_json, result_json, ef, sL, sH, svL, svH, pv, emb):
    if not result_json:
        return no_update, None, None

    btn = ctx.triggered_id
    df  = pd.read_json(result_json, orient="split")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": now,
        "parameters": {
            "EF (kg/kWh)": ef,
            "Share Light": sL,
            "Share HVAC": sH,
            "Saving Light": svL,
            "Saving HVAC": svH,
            "PV offset": pv,
            "Embodied (t)": emb
        }
    }

    if btn == "save_btn":
        outdir = ensure_dir("./exports")
        df.to_csv(os.path.join(outdir, f"scenario_results_{now}.csv"), index=False)
        with open(os.path.join(outdir, f"scenario_results_{now}.json"), "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": df.to_dict(orient="records")}, f, ensure_ascii=False, indent=2)
        return "Results saved to ./exports (CSV & JSON).", None, None

    if btn == "dl_csv_btn":
        return no_update, dcc.send_data_frame(df.to_csv, f"scenario_results_{now}.csv", index=False), None

    if btn == "dl_json_btn":
        payload = {"meta": meta, "results": df.to_dict(orient="records")}
        return no_update, None, dict(content=json.dumps(payload, ensure_ascii=False, indent=2),
                                     filename=f"scenario_results_{now}.json")

    return no_update, None, None

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
