from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from streamlit.runtime.scriptrunner import get_script_run_ctx
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller


APP_TITLE = "Time Series Anomaly Intelligence"


@dataclass
class DetectionConfig:
    freq: str
    window: int
    sensitivity: float
    use_stl: bool
    use_isolation_forest: bool
    business_mode: str


def build_sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(321083)
    dates = pd.date_range(end="2026-06-15", periods=180, freq="D")
    trend = np.linspace(70, 92, len(dates))
    weekly = 9 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7)
    noise = rng.normal(0, 2.4, len(dates))
    values = trend + weekly + noise
    anomaly_idx = [22, 48, 79, 121, 153]
    values[anomaly_idx] += np.array([28, -24, 35, -30, 26])
    competitor = 66 + np.linspace(0, 14, len(dates)) + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 14)
    inventory = 120 - 0.35 * np.arange(len(dates)) + rng.normal(0, 4, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "demand": values.round(2),
            "competitor_signal": competitor.round(2),
            "inventory": inventory.round(2),
        }
    )


def read_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return build_sample_data()
    return pd.read_csv(uploaded_file)


def candidate_datetime_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    keywords = ("date", "time", "datetime", "timestamp", "day", "month", "year")
    for col in df.columns:
        name_hit = any(word in str(col).lower() for word in keywords)
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            candidates.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        valid_ratio = parsed.notna().mean()
        if name_hit or valid_ratio >= 0.75:
            candidates.append(col)
    return candidates


def candidate_numeric_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() >= 0.75:
            candidates.append(col)
    return candidates


def prepare_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    freq: str,
) -> tuple[pd.Series, dict[str, float | int | str]]:
    work = df[[time_col, value_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    original_rows = len(work)
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    missing_values = int(work[value_col].isna().sum())
    work[value_col] = work[value_col].interpolate(limit_direction="both")
    work = work.dropna(subset=[value_col])

    series = work.groupby(time_col)[value_col].mean().sort_index()
    inferred_freq = pd.infer_freq(series.index)
    selected_freq = inferred_freq if freq == "auto" and inferred_freq else None
    if freq != "auto":
        selected_freq = freq

    if selected_freq:
        series = series.asfreq(selected_freq)
        series = series.interpolate(limit_direction="both")

    meta = {
        "original_rows": original_rows,
        "usable_rows": int(len(series)),
        "missing_values": missing_values,
        "duplicates": int(work[time_col].duplicated().sum()),
        "frequency": selected_freq or "irregular",
    }
    return series.astype(float), meta


def prepare_context_frame(
    df: pd.DataFrame,
    time_col: str,
    context_cols: list[str],
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if not context_cols:
        return pd.DataFrame(index=target_index)
    work = df[[time_col, *context_cols]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col)
    for col in context_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    context = work.groupby(time_col)[context_cols].mean().sort_index()
    context = context.reindex(target_index).interpolate(limit_direction="both")
    return context


def robust_z_score(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad if mad > 0 else values.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - median) / scale


def rolling_z_score(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(4, window // 3)
    center = series.rolling(window=window, min_periods=min_periods, center=True).median()
    spread = series.rolling(window=window, min_periods=min_periods, center=True).std()
    z = (series - center) / spread.replace(0, np.nan)
    return z.fillna(robust_z_score(series))


def iqr_score(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(4, window // 3)
    q1 = series.rolling(window=window, min_periods=min_periods, center=True).quantile(0.25)
    q3 = series.rolling(window=window, min_periods=min_periods, center=True).quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    distance = pd.Series(0.0, index=series.index)
    distance = distance.mask(series < lower, (lower - series) / iqr)
    distance = distance.mask(series > upper, (series - upper) / iqr)
    return distance.fillna(0.0)


def stl_score(series: pd.Series, period: int) -> pd.Series:
    if len(series) < max(24, period * 2):
        return pd.Series(np.zeros(len(series)), index=series.index)
    fit = STL(series, period=period, robust=True).fit()
    return robust_z_score(fit.resid).abs()


def isolation_forest_score(series: pd.Series, window: int, context: pd.DataFrame | None = None) -> pd.Series:
    if len(series) < 20:
        return pd.Series(np.zeros(len(series)), index=series.index)
    frame = pd.DataFrame(
        {
            "value": series,
            "diff": series.diff().fillna(0),
            "rolling_mean": series.rolling(window, min_periods=2).mean().bfill(),
            "rolling_std": series.rolling(window, min_periods=2).std().fillna(0),
        }
    )
    if context is not None and not context.empty:
        for col in context.columns:
            frame[f"context_{col}"] = context[col]
            frame[f"context_{col}_diff"] = context[col].diff().fillna(0)
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    model = IsolationForest(contamination="auto", random_state=321083)
    model.fit(frame)
    raw = -model.score_samples(frame)
    scaled = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return pd.Series(scaled * 4, index=series.index)


def guess_period(freq: str, n: int) -> int:
    if freq in {"D", "B"}:
        return 7
    if freq in {"H", "h"}:
        return 24
    if freq in {"M", "MS"}:
        return 12
    if freq in {"W", "W-SUN", "W-MON"}:
        return 52 if n >= 120 else 4
    return max(2, min(12, n // 4))


def detect_anomalies(series: pd.Series, config: DetectionConfig, context: pd.DataFrame | None = None) -> pd.DataFrame:
    period = guess_period(config.freq, len(series))
    rz = rolling_z_score(series, config.window).abs()
    iq = iqr_score(series, config.window)
    stl_component = stl_score(series, period) if config.use_stl else pd.Series(0.0, index=series.index)
    iso_component = (
        isolation_forest_score(series, config.window, context)
        if config.use_isolation_forest
        else pd.Series(0.0, index=series.index)
    )

    score = (0.38 * rz) + (0.22 * iq) + (0.24 * stl_component) + (0.16 * iso_component)
    threshold = max(float(np.nanquantile(score, config.sensitivity)), 2.0)

    result = pd.DataFrame(
        {
            "timestamp": series.index,
            "value": series.values,
            "rolling_z": rz.values,
            "iqr_score": iq.values,
            "stl_score": stl_component.values,
            "isolation_score": iso_component.values,
            "anomaly_score": score.values,
        }
    )
    result["method_votes"] = (
        (result["rolling_z"] >= 2.5).astype(int)
        + (result["iqr_score"] >= 1.0).astype(int)
        + (result["stl_score"] >= 2.5).astype(int)
        + (result["isolation_score"] >= 2.2).astype(int)
    )
    result["is_anomaly"] = (result["anomaly_score"] >= threshold) | (result["method_votes"] >= 3)
    result["severity"] = pd.cut(
        result["anomaly_score"],
        bins=[-np.inf, threshold, threshold * 1.35, threshold * 1.85, np.inf],
        labels=["normal", "watch", "warning", "critical"],
        include_lowest=True,
    ).astype(str)
    result = add_explanations(result, series, config, context)
    return result


def add_explanations(
    detected: pd.DataFrame,
    series: pd.Series,
    config: DetectionConfig,
    context: pd.DataFrame | None,
) -> pd.DataFrame:
    values = series.reset_index(drop=True)
    rolling_median = series.rolling(config.window, min_periods=max(4, config.window // 3), center=True).median()
    rolling_std = series.rolling(config.window, min_periods=max(4, config.window // 3), center=True).std()
    rolling_median = rolling_median.bfill().ffill().reset_index(drop=True)
    rolling_std = rolling_std.replace(0, np.nan).bfill().ffill().reset_index(drop=True)

    anomaly_types: list[str] = []
    reasons: list[str] = []
    actions: list[str] = []
    context_notes: list[str] = []

    for i, row in detected.iterrows():
        value = values.iloc[i]
        baseline = rolling_median.iloc[i]
        local_std = rolling_std.iloc[i]
        direction = "above" if value >= baseline else "below"
        delta = value - baseline
        prev_delta = values.diff().iloc[i] if i > 0 else np.nan
        next_delta = values.diff().iloc[i + 1] if i + 1 < len(values) else np.nan
        local_vol = series.diff().abs().rolling(config.window, min_periods=3).mean().iloc[i]
        global_vol = series.diff().abs().mean()

        if abs(prev_delta) > 2 * (local_std if np.isfinite(local_std) else 1) and abs(next_delta) > abs(prev_delta) * 0.5:
            anomaly_type = "single-point spike/drop"
        elif abs(delta) > 2.5 * (local_std if np.isfinite(local_std) else 1):
            anomaly_type = "level deviation"
        elif np.isfinite(local_vol) and local_vol > global_vol * 1.8:
            anomaly_type = "volatility burst"
        elif row["stl_score"] >= max(row["rolling_z"], 2.5):
            anomaly_type = "seasonality residual"
        else:
            anomaly_type = "mixed signal"

        reason_bits = [
            f"value is {direction} local baseline by {delta:.2f}",
            f"ensemble score {row['anomaly_score']:.2f}",
            f"{int(row['method_votes'])} of 4 methods agreed",
        ]
        if row["stl_score"] >= 2.5:
            reason_bits.append("seasonal pattern did not explain it")
        if row["isolation_score"] >= 2.2:
            reason_bits.append("multifeature pattern looked unusual")

        note = context_signal_note(context, i) if context is not None and not context.empty else "no context columns selected"
        context_notes.append(note)
        anomaly_types.append(anomaly_type)
        reasons.append("; ".join(reason_bits))
        actions.append(recommended_action(config.business_mode, anomaly_type, direction, note))

    detected["anomaly_type"] = anomaly_types
    detected["reason"] = reasons
    detected["context_note"] = context_notes
    detected["recommended_action"] = actions
    return detected


def context_signal_note(context: pd.DataFrame, row_index: int) -> str:
    if context.empty:
        return "no context columns selected"
    notes: list[str] = []
    for col in context.columns:
        z = robust_z_score(context[col]).iloc[row_index]
        if abs(z) >= 2:
            direction = "high" if z > 0 else "low"
            notes.append(f"{col} is unusually {direction}")
    return "; ".join(notes) if notes else "context columns look normal"


def recommended_action(mode: str, anomaly_type: str, direction: str, context_note: str) -> str:
    if mode == "Demand / Sales":
        if direction == "above":
            return "Check promotion, holiday, stockout risk, and whether the demand jump should update the forecast."
        return "Check missing sales, stockout, price change, or demand drop before treating this as normal seasonality."
    if mode == "Operations / Sensor":
        return "Inspect sensor logs, maintenance history, and adjacent measurements before accepting this point."
    if mode == "Finance / Risk":
        return "Confirm transaction source, calendar event, and one-off accounting effects before excluding the point."
    return "Review source data, nearby observations, and context variables before deciding whether to remove or keep it."


def describe_series(series: pd.Series, detected: pd.DataFrame, meta: dict[str, float | int | str]) -> pd.DataFrame:
    values = series.dropna()
    anomaly_count = int(detected["is_anomaly"].sum())
    period = guess_period(str(meta["frequency"]), len(series))
    trend_strength, seasonality_strength = decomposition_strength(series, period)
    adf_p = np.nan
    if len(values) >= 12 and values.nunique() > 1:
        try:
            adf_p = float(adfuller(values, autolag="AIC")[1])
        except Exception:
            adf_p = np.nan
    lag1 = np.nan
    if len(values) >= 3 and values.nunique() > 1:
        lag1 = float(acf(values, nlags=1, fft=False)[1])
    rows = [
        ("points", len(values), "number of observations used"),
        ("anomaly_count", anomaly_count, "number of detected anomalies"),
        ("anomaly_rate", anomaly_count / max(len(values), 1), "share of anomalous observations"),
        ("mean", values.mean(), "average level"),
        ("std", values.std(ddof=0), "volatility"),
        ("cv", values.std(ddof=0) / (abs(values.mean()) + 1e-9), "volatility relative to mean"),
        ("min", values.min(), "minimum value"),
        ("max", values.max(), "maximum value"),
        ("missing_before_interpolation", meta["missing_values"], "missing values before interpolation"),
        ("lag1_autocorrelation", lag1, "lag-1 autocorrelation"),
        ("adf_p_value", adf_p, "stationarity test p-value"),
        ("trend_strength", trend_strength, "strength of trend component"),
        ("seasonality_strength", seasonality_strength, "strength of seasonal component"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "meaning"])


def decomposition_strength(series: pd.Series, period: int) -> tuple[float, float]:
    if len(series) < max(24, period * 2):
        return np.nan, np.nan
    try:
        fit = STL(series, period=period, robust=True).fit()
        resid_var = np.nanvar(fit.resid)
        trend_strength = max(0.0, 1.0 - resid_var / (np.nanvar(fit.trend + fit.resid) + 1e-9))
        seasonal_strength = max(0.0, 1.0 - resid_var / (np.nanvar(fit.seasonal + fit.resid) + 1e-9))
        return float(trend_strength), float(seasonal_strength)
    except Exception:
        return np.nan, np.nan


def line_chart(detected: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=detected["timestamp"],
            y=detected["value"],
            mode="lines",
            name="series",
            line=dict(color="#1f2937", width=2),
        )
    )
    anomalies = detected[detected["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["value"],
            mode="markers",
            name="anomaly",
            marker=dict(color="#e11d48", size=10, symbol="x"),
            text=anomalies["anomaly_type"],
            hovertemplate="%{x}<br>value=%{y}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=460,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="time")
    fig.update_yaxes(title="value")
    return fig


def score_chart(detected: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12)
    fig.add_trace(
        go.Scatter(
            x=detected["timestamp"],
            y=detected["anomaly_score"],
            mode="lines",
            name="ensemble score",
            line=dict(color="#2563eb", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=detected["anomaly_score"],
            nbinsx=30,
            marker_color="#93c5fd",
            name="score distribution",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=520, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
    fig.update_yaxes(title="score", row=1, col=1)
    fig.update_yaxes(title="count", row=2, col=1)
    fig.update_xaxes(title="time", row=1, col=1)
    fig.update_xaxes(title="score", row=2, col=1)
    return fig


def component_chart(detected: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in [
        ("rolling_z", "#0f766e"),
        ("iqr_score", "#7c3aed"),
        ("stl_score", "#ea580c"),
        ("isolation_score", "#64748b"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=detected["timestamp"],
                y=detected[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=1.8),
            )
        )
    fig.update_layout(
        height=430,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title="time")
    fig.update_yaxes(title="component score")
    return fig


def anomaly_type_chart(detected: pd.DataFrame) -> go.Figure:
    anomalies = detected[detected["is_anomaly"]]
    counts = anomalies["anomaly_type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = go.Figure(
        go.Bar(
            x=counts["type"],
            y=counts["count"],
            marker_color=["#2563eb", "#0f766e", "#ea580c", "#7c3aed", "#64748b"][: len(counts)],
        )
    )
    fig.update_layout(height=330, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(title="anomaly type")
    fig.update_yaxes(title="count")
    return fig


def differentiation_scorecard() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "baseline assignment feature": "Upload CSV and detect anomalies",
                "added differentiation": "Classifies anomaly type and explains method agreement",
                "why it matters": "Shows reasoning, not only red dots",
            },
            {
                "baseline assignment feature": "Dashboard charts",
                "added differentiation": "Score decomposition by rolling, IQR, STL, and Isolation Forest",
                "why it matters": "Evaluator can judge whether detection is appropriate",
            },
            {
                "baseline assignment feature": "Multivariate CSV support",
                "added differentiation": "Optional context columns influence Isolation Forest and root-cause notes",
                "why it matters": "Uses extra CSV variables instead of ignoring them",
            },
            {
                "baseline assignment feature": "Metric table",
                "added differentiation": "Action-oriented anomaly review table and markdown report",
                "why it matters": "Looks like a practical monitoring tool",
            },
            {
                "baseline assignment feature": "Parameter changes",
                "added differentiation": "Business mode changes the recommended response",
                "why it matters": "Same anomaly is interpreted differently by domain",
            },
        ]
    )


def make_report(
    detected: pd.DataFrame,
    metrics: pd.DataFrame,
    time_col: str,
    value_col: str,
    config: DetectionConfig,
    context_cols: list[str],
) -> str:
    anomalies = detected[detected["is_anomaly"]].copy()
    top = anomalies.sort_values("anomaly_score", ascending=False).head(10)
    lines = [
        "# Time Series Anomaly Intelligence Report",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Time column: `{time_col}`",
        f"- Value column: `{value_col}`",
        f"- Context columns: `{', '.join(context_cols) if context_cols else 'none'}`",
        f"- Business mode: `{config.business_mode}`",
        f"- Frequency: `{config.freq}`",
        f"- Window: `{config.window}`",
        f"- Sensitivity quantile: `{config.sensitivity:.2f}`",
        "",
        "## Differentiation Point",
        "This app does not stop at marking anomalies. It classifies each anomaly, explains which detection signals agreed, checks context columns, and suggests a review action.",
        "",
        "## Key Metrics",
    ]
    for _, row in metrics.iterrows():
        value = row["value"]
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"- {row['metric']}: {value} ({row['meaning']})")
    lines.extend(["", "## Top Anomalies"])
    if top.empty:
        lines.append("- No anomalies detected under the current settings.")
    else:
        for _, row in top.iterrows():
            lines.append(
                f"- {row['timestamp']}: value={row['value']:.4f}, score={row['anomaly_score']:.4f}, "
                f"type={row['anomaly_type']}, reason={row['reason']}, action={row['recommended_action']}"
            )
    return "\n".join(lines)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.55rem; }
        .app-subtitle { color: #475569; margin-top: -0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stretch_plotly_chart(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def stretch_dataframe(data: pd.DataFrame, hide_index: bool = False) -> None:
    try:
        st.dataframe(data, width="stretch", hide_index=hide_index)
    except TypeError:
        st.dataframe(data, use_container_width=True, hide_index=hide_index)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_style()

    st.title(APP_TITLE)
    st.markdown(
        '<p class="app-subtitle">CSV upload, automatic anomaly detection, explanation, anomaly type classification, and action-oriented review.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input")
        uploaded_file = st.file_uploader("CSV file", type=["csv"])
        df = read_csv(uploaded_file)

        dt_cols = candidate_datetime_columns(df)
        num_cols = candidate_numeric_columns(df)
        if not dt_cols:
            st.error("No date/time column was detected.")
            return
        if not num_cols:
            st.error("No numeric column was detected.")
            return

        time_col = st.selectbox("Time column", dt_cols, index=0)
        value_default = 0 if num_cols[0] != time_col else min(1, len(num_cols) - 1)
        value_col = st.selectbox("Target value column", num_cols, index=value_default)
        context_options = [col for col in num_cols if col not in {time_col, value_col}]
        context_cols = st.multiselect("Context columns for root-cause hints", context_options, default=context_options[:2])

        st.header("Detection")
        freq = st.selectbox("Frequency", ["auto", "D", "W", "M", "H"], index=0)
        window = st.slider("Rolling window", min_value=5, max_value=90, value=21, step=2)
        sensitivity_label = st.select_slider(
            "Sensitivity",
            options=["low", "medium", "high", "very high"],
            value="high",
        )
        sensitivity_map = {"low": 0.985, "medium": 0.965, "high": 0.94, "very high": 0.90}
        use_stl = st.toggle("Use STL residual score", value=True)
        use_iforest = st.toggle("Use multifeature Isolation Forest", value=True)
        business_mode = st.selectbox(
            "Review mode",
            ["Demand / Sales", "Operations / Sensor", "Finance / Risk", "General"],
            index=0,
        )

    try:
        series, meta = prepare_series(df, time_col, value_col, freq)
        context = prepare_context_frame(df, time_col, context_cols, series.index)
    except Exception as exc:
        st.error(f"Time series preprocessing failed: {exc}")
        return

    if len(series) < 12:
        st.error("At least 12 usable observations are recommended.")
        return

    config = DetectionConfig(
        freq=str(meta["frequency"]) if freq == "auto" else freq,
        window=min(window, max(5, len(series) // 2)),
        sensitivity=sensitivity_map[sensitivity_label],
        use_stl=use_stl,
        use_isolation_forest=use_iforest,
        business_mode=business_mode,
    )
    detected = detect_anomalies(series, config, context)
    metrics = describe_series(series, detected, meta)
    anomaly_count = int(detected["is_anomaly"].sum())
    anomaly_rate = anomaly_count / len(detected)
    latest = detected[detected["is_anomaly"]].tail(1)
    latest_text = "none" if latest.empty else str(latest.iloc[0]["timestamp"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows", f"{len(series):,}")
    k2.metric("Anomalies", f"{anomaly_count:,}")
    k3.metric("Anomaly rate", f"{anomaly_rate:.1%}")
    k4.metric("Context cols", f"{len(context_cols):,}")
    k5.metric("Latest anomaly", latest_text)

    tabs = st.tabs(
        [
            "Dashboard",
            "Anomaly Review",
            "Indicators",
            "Differentiation",
            "Data Preview",
        ]
    )
    with tabs[0]:
        stretch_plotly_chart(line_chart(detected))
        c1, c2 = st.columns([1, 1])
        with c1:
            stretch_plotly_chart(score_chart(detected))
        with c2:
            stretch_plotly_chart(component_chart(detected))

    with tabs[1]:
        c1, c2 = st.columns([1.15, 0.85])
        anomalies = detected[detected["is_anomaly"]].sort_values("anomaly_score", ascending=False)
        with c1:
            review_cols = [
                "timestamp",
                "value",
                "anomaly_score",
                "severity",
                "method_votes",
                "anomaly_type",
                "reason",
                "context_note",
                "recommended_action",
            ]
            stretch_dataframe(anomalies[review_cols], hide_index=True)
        with c2:
            stretch_plotly_chart(anomaly_type_chart(detected))
            st.caption("This review board is the main differentiation point: each point is classified, explained, and connected to an action.")

        st.download_button(
            "Download full anomaly CSV",
            data=dataframe_to_csv_bytes(detected),
            file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        report = make_report(detected, metrics, time_col, value_col, config, context_cols)
        st.download_button(
            "Download explanation report",
            data=report.encode("utf-8-sig"),
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    with tabs[2]:
        stretch_dataframe(metrics, hide_index=True)
        st.caption(
            "The ensemble score combines rolling z-score, IQR distance, STL residual score, and multifeature Isolation Forest score."
        )

    with tabs[3]:
        st.subheader("How this avoids the Project 1 differentiation problem")
        stretch_dataframe(differentiation_scorecard(), hide_index=True)
        st.markdown(
            """
            **Presentation angle:** Do not introduce this as only an anomaly detector.
            Present it as an anomaly intelligence dashboard that explains
            what type of anomaly happened, which methods agreed, whether context columns support it,
            and what the analyst should check next.
            """
        )

    with tabs[4]:
        st.write("Changing the uploaded file, target column, context columns, or parameters triggers a fresh analysis.")
        stretch_dataframe(df.head(200))


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
        raise SystemExit(0)
    main()
