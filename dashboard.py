"""
Credit Risk Validation Dashboard
---------------------------------
Reads from validation_log.db and displays run history, DQ findings,
and model performance results.

Run:  python dashboard.py
Open: http://127.0.0.1:8051
"""

import sqlite3

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

DB_PATH = "validation_log.db"

# ── data helpers ──────────────────────────────────────────────────────────────

def _read(sql: str, params=None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(sql, conn, params=params or [])

def load_runs()           -> pd.DataFrame: return _read("SELECT * FROM validation_runs ORDER BY timestamp DESC")
def load_findings(run_id) -> pd.DataFrame: return _read("SELECT * FROM dq_findings WHERE run_id = ?",           [run_id])
def load_metrics(run_id)  -> pd.DataFrame: return _read("SELECT * FROM model_metrics WHERE run_id = ?",          [run_id])
def load_benchmark(run_id)-> pd.DataFrame: return _read("SELECT * FROM benchmark_comparisons WHERE run_id = ?",  [run_id])

# ── colour maps ───────────────────────────────────────────────────────────────

_STATUS_BS   = {"PASS": "success", "FAIL": "danger"}
_SEVERITY_BS = {"CRITICAL": "danger", "HIGH": "warning", "MEDIUM": "warning", "LOW": "info", "INFO": "secondary"}
_SEVERITY_HEX = {"CRITICAL": "#dc3545", "HIGH": "#fd7e14", "MEDIUM": "#ffc107", "LOW": "#0dcaf0", "INFO": "#adb5bd"}

# ── UI helpers ────────────────────────────────────────────────────────────────

def _kpi(title: str, value: str, color: str = "primary") -> dbc.Col:
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.8rem"}),
            html.H4(value, className=f"text-{color} fw-bold mb-0"),
        ]), className="shadow-sm h-100"),
    )

def _run_options(runs: pd.DataFrame) -> list:
    return [
        {"label": f"{r['dataset_name']}  |  {r['timestamp'][:19]}", "value": r["run_id"]}
        for _, r in runs.iterrows()
    ]

# ── app ───────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Validation Dashboard",
)
server = app.server

# ── layout ────────────────────────────────────────────────────────────────────

_initial_runs = load_runs()
_initial_opts = _run_options(_initial_runs)
_initial_run  = _initial_opts[0]["value"] if _initial_opts else None

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(
            html.H3("Credit Risk Validation Dashboard", className="text-primary fw-bold mb-0"),
            width="auto",
        ),
        dbc.Col(
            dbc.Badge("Model Risk · IFRS 9", color="secondary", className="ms-2 align-self-center"),
            width="auto",
        ),
        dbc.Col(
            dbc.Button("↻ Refresh", id="refresh-btn", color="outline-primary",
                       size="sm", className="ms-auto"),
            width="auto",
        ),
    ], align="center", className="py-3 border-bottom"),

    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Run History",       tab_id="tab-history"),
        dbc.Tab(label="Data Quality Results", tab_id="tab-dq"),
        dbc.Tab(label="Model Performance", tab_id="tab-perf"),
    ], id="main-tabs", active_tab="tab-history", className="mt-3"),

    # Run selector — hidden on Tab 1, shown on Tabs 2 and 3
    dbc.Row(
        dbc.Col([
            html.Label("Validation Run", className="fw-bold mt-3 mb-1 small text-muted"),
            dcc.Dropdown(
                id="run-selector",
                options=_initial_opts,
                value=_initial_run,
                clearable=False,
            ),
        ], width=6),
        id="run-selector-row",
        style={"display": "none"},
    ),

    html.Div(id="tab-content", className="mt-3"),

    dcc.Store(id="runs-store", data=_initial_runs.to_dict("records")),
    dcc.Interval(id="auto-refresh", interval=30_000, n_intervals=0),
], fluid=True)

# ── callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("runs-store",       "data"),
    Output("run-selector",     "options"),
    Input("auto-refresh",      "n_intervals"),
    Input("refresh-btn",       "n_clicks"),
)
def refresh_store(_, __):
    runs = load_runs()
    return runs.to_dict("records"), _run_options(runs)


@callback(
    Output("run-selector-row", "style"),
    Input("main-tabs",         "active_tab"),
)
def toggle_selector(tab):
    return {"display": "none"} if tab == "tab-history" else {"marginTop": "12px"}


@callback(
    Output("tab-content",  "children"),
    Input("main-tabs",     "active_tab"),
    Input("run-selector",  "value"),
    Input("runs-store",    "data"),
)
def render_tab(active_tab, run_id, runs_data):
    runs = pd.DataFrame(runs_data or [])
    if active_tab == "tab-history":
        return _tab_history(runs)
    if active_tab == "tab-dq":
        return _tab_dq(run_id)
    if active_tab == "tab-perf":
        return _tab_perf(run_id)
    return html.P("Select a tab.")


# ── Tab 1: Run History ────────────────────────────────────────────────────────

def _tab_history(runs: pd.DataFrame):
    if runs.empty:
        return dbc.Alert(
            "No validation runs found. Run  python run_validation.py  first.",
            color="warning",
        )

    n_pass  = int((runs["status"] == "PASS").sum())
    n_fail  = int((runs["status"] == "FAIL").sum())
    n_abort = int(runs["status"].str.startswith("ABORTED").sum())

    donut = go.Figure(go.Pie(
        labels=["PASS", "FAIL", "ABORTED"],
        values=[n_pass, n_fail, n_abort],
        hole=0.55,
        marker_colors=["#198754", "#dc3545", "#fd7e14"],
        textinfo="label+value",
    ))
    donut.update_layout(
        showlegend=False, height=300,
        margin=dict(t=40, b=0, l=0, r=0),
        title_text="Status Distribution",
    )

    rows = []
    for _, r in runs.iterrows():
        color = "success" if r["status"] == "PASS" else \
                "danger"  if r["status"] == "FAIL" else "warning"
        rows.append(html.Tr([
            html.Td(r["run_id"],
                    style={"fontSize": "0.72rem", "fontFamily": "monospace", "whiteSpace": "nowrap"}),
            html.Td(r["dataset_name"]),
            html.Td(r["model_name"]),
            html.Td(dbc.Badge(r["status"], color=color)),
            html.Td(f"{int(r['n_samples']):,}"),
            html.Td(int(r["n_findings"])),
            html.Td(r["timestamp"][:19], style={"fontSize": "0.8rem"}),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Run ID"), html.Th("Dataset"), html.Th("Model"),
            html.Th("Status"), html.Th("Rows"), html.Th("Findings"), html.Th("Timestamp (UTC)"),
        ])),
        html.Tbody(rows),
    ], striped=True, bordered=True, hover=True, responsive=True, size="sm", className="mb-0")

    return html.Div([
        dbc.Row([
            _kpi("Total Runs", str(len(runs)), "primary"),
            _kpi("Pass",       str(n_pass),    "success"),
            _kpi("Fail",       str(n_fail),    "danger"),
            _kpi("Aborted",    str(n_abort),   "warning"),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(table,                   width=8),
            dbc.Col(dcc.Graph(figure=donut), width=4),
        ]),
    ])


# ── Tab 2: DQ Findings ────────────────────────────────────────────────────────

def _tab_dq(run_id):
    if not run_id:
        return dbc.Alert("Select a validation run from the dropdown above.", color="info")

    findings = load_findings(run_id)
    if findings.empty:
        return dbc.Alert("No DQ findings recorded for this run.", color="info")

    sev_order  = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    sev_counts = (
        findings.groupby("severity").size()
        .reindex(sev_order).fillna(0).astype(int).reset_index(name="count")
    )
    sev_counts = sev_counts[sev_counts["count"] > 0]

    bar = go.Figure(go.Bar(
        x=sev_counts["severity"],
        y=sev_counts["count"],
        marker_color=[_SEVERITY_HEX.get(s, "#adb5bd") for s in sev_counts["severity"]],
        text=sev_counts["count"],
        textposition="outside",
    ))
    bar.update_layout(
        title="Findings by Severity", height=280,
        yaxis_title="Count", margin=dict(t=40, b=20),
        yaxis=dict(dtick=1),
    )

    rows = []
    for _, f in findings.iterrows():
        rows.append(html.Tr([
            html.Td(f["check_name"]),
            html.Td(dbc.Badge(
                f["status"],
                color="success" if f["status"] == "PASS" else
                       "warning" if f["status"] == "WARN" else "danger",
            )),
            html.Td(dbc.Badge(f["severity"], color=_SEVERITY_BS.get(f["severity"], "secondary"))),
            html.Td(str(int(f["affected_rows"]))),
            html.Td(f["detail"], style={"fontSize": "0.8rem"}),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Check"), html.Th("Status"), html.Th("Severity"),
            html.Th("Affected Rows"), html.Th("Detail"),
        ])),
        html.Tbody(rows),
    ], striped=True, bordered=True, hover=True, responsive=True, size="sm")

    return dbc.Row([
        dbc.Col(table,                  width=8),
        dbc.Col(dcc.Graph(figure=bar),  width=4),
    ])


# ── Tab 3: Model Performance ──────────────────────────────────────────────────

def _tab_perf(run_id):
    if not run_id:
        return dbc.Alert("Select a validation run from the dropdown above.", color="info")

    metrics   = load_metrics(run_id)
    benchmark = load_benchmark(run_id)
    run_info  = _read("SELECT * FROM validation_runs WHERE run_id = ?", [run_id])
    status    = run_info.iloc[0]["status"] if not run_info.empty else "N/A"
    color     = "success" if status == "PASS" else "danger" if status == "FAIL" else "warning"

    if metrics.empty:
        return dbc.Alert(
            [html.Strong("No model metrics for this run. "),
             f"Status: {status}. This run was aborted at the schema gate before scoring."],
            color="warning",
        )

    avg_f1   = float(metrics["macro_f1"].mean())
    avg_rec  = float(metrics["recall"].mean())
    total_n  = int(metrics["n_samples"].sum())
    dr_pct   = f"{float(metrics['default_rate'].mean()):.1%}"

    # Segment grouped bar
    seg_fig = go.Figure()
    for metric, label, clr in [
        ("macro_f1",  "Macro F1",  "#0d6efd"),
        ("precision", "Precision", "#198754"),
        ("recall",    "Recall",    "#fd7e14"),
    ]:
        seg_fig.add_trace(go.Bar(
            name=label,
            x=metrics["segment"],
            y=metrics[metric],
            marker_color=clr,
            text=metrics[metric].map("{:.3f}".format),
            textposition="outside",
        ))
    seg_fig.update_layout(
        barmode="group",
        title="Segment-Level Performance",
        yaxis=dict(range=[0, 1.15], title="Score"),
        height=340,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", y=-0.15),
    )

    # Benchmark horizontal bar
    if not benchmark.empty:
        bench_fig = go.Figure(go.Bar(
            x=benchmark["macro_f1"],
            y=benchmark["model_name"],
            orientation="h",
            marker_color=["#0d6efd", "#198754", "#adb5bd"],
            text=benchmark["macro_f1"].map("{:.4f}".format),
            textposition="outside",
        ))
        bench_fig.update_layout(
            title="Benchmark Comparison — Macro F1",
            xaxis=dict(range=[0, 1.15], title="Macro F1"),
            height=280,
            margin=dict(t=40, b=20),
        )
        bench_el = dcc.Graph(figure=bench_fig)
    else:
        bench_el = dbc.Alert("No benchmark data for this run.", color="info")

    return html.Div([
        dbc.Row([
            _kpi("Status",        status,          color),
            _kpi("Scored Rows",   f"{total_n:,}",  "primary"),
            _kpi("Avg Macro F1",  f"{avg_f1:.4f}", "success" if avg_f1 >= 0.52 else "danger"),
            _kpi("Avg Recall",    f"{avg_rec:.4f}", "info"),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=seg_fig), width=7),
            dbc.Col(bench_el,                  width=5),
        ]),
    ])


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8051)
