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
from statsmodels.tsa.stattools import acf, adfuller, pacf


APP_TITLE = "시계열 이상탐지 분석 대시보드"
THEME = {
    "ink": "#242331",
    "muted": "#6f7f7b",
    "paper": "#fbfbf8",
    "panel": "#ffffff",
    "mint": "#dfeeea",
    "mint_strong": "#b7dfa8",
    "coral": "#e78c73",
    "coral_dark": "#c96953",
    "line": "#2c2b3b",
    "grid": "#d8e3df",
}


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
        labels=["정상", "관찰", "주의", "위험"],
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
            anomaly_type = "단일 시점 급등/급락"
        elif abs(delta) > 2.5 * (local_std if np.isfinite(local_std) else 1):
            anomaly_type = "수준 이탈"
        elif np.isfinite(local_vol) and local_vol > global_vol * 1.8:
            anomaly_type = "변동성 급증"
        elif row["stl_score"] >= max(row["rolling_z"], 2.5):
            anomaly_type = "계절성 잔차 이상"
        else:
            anomaly_type = "복합 신호"

        reason_bits = [
            f"국소 기준선 대비 {'높음' if direction == 'above' else '낮음'}: {delta:.2f}",
            f"종합 이상 점수 {row['anomaly_score']:.2f}",
            f"4개 기준 중 {int(row['method_votes'])}개가 이상 신호로 판단",
        ]
        if row["stl_score"] >= 2.5:
            reason_bits.append("반복 계절 패턴만으로 설명하기 어려움")
        if row["isolation_score"] >= 2.2:
            reason_bits.append("다변량 패턴 관점에서도 비정상적")

        note = context_signal_note(context, i) if context is not None and not context.empty else "참고 변수 미선택"
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
            direction = "높음" if z > 0 else "낮음"
            notes.append(f"{col} 값이 평소보다 {direction}")
    return "; ".join(notes) if notes else "참고 변수에서 특이 신호 없음"


def recommended_action(mode: str, anomaly_type: str, direction: str, context_note: str) -> str:
    if mode in {"Demand / Sales", "수요/매출"}:
        if direction == "above":
            return "프로모션, 휴일 효과, 재고 부족 가능성, 예측 기준 상향 필요성을 확인하세요."
        return "매출 누락, 재고 부족, 가격 변경, 수요 감소 요인을 확인한 뒤 정상 계절성인지 판단하세요."
    if mode in {"Operations / Sensor", "운영/센서"}:
        return "센서 로그, 유지보수 이력, 인접 측정값을 함께 확인한 뒤 이상 여부를 확정하세요."
    if mode in {"Finance / Risk", "금융/리스크"}:
        return "거래 출처, 달력 이벤트, 일회성 회계 효과를 확인한 뒤 제외 여부를 판단하세요."
    return "원천 데이터, 주변 관측치, 참고 변수를 함께 검토한 뒤 제거/유지 여부를 결정하세요."


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
        ("관측치 수", len(values), "분석에 사용된 시계열 관측치 개수"),
        ("이상치 수", anomaly_count, "탐지된 이상치 개수"),
        ("이상치 비율", anomaly_count / max(len(values), 1), "전체 관측치 중 이상치 비율"),
        ("평균", values.mean(), "시계열의 평균 수준"),
        ("표준편차", values.std(ddof=0), "값의 변동성"),
        ("변동계수", values.std(ddof=0) / (abs(values.mean()) + 1e-9), "평균 대비 상대 변동성"),
        ("최솟값", values.min(), "관측값의 최솟값"),
        ("최댓값", values.max(), "관측값의 최댓값"),
        ("보간 전 결측치", meta["missing_values"], "선형 보간 전 결측값 개수"),
        ("1시차 자기상관", lag1, "직전 시점과 현재 시점의 상관 정도"),
        ("ADF p-value", adf_p, "정상성 검정 p-value"),
        ("추세 강도", trend_strength, "추세 성분의 상대적 강도"),
        ("계절성 강도", seasonality_strength, "계절 성분의 상대적 강도"),
    ]
    return pd.DataFrame(rows, columns=["지표", "값", "의미"])


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    match = metrics.loc[metrics["지표"] == name, "값"]
    if match.empty:
        return np.nan
    return float(match.iloc[0])


def generate_analysis_briefing(
    series: pd.Series,
    detected: pd.DataFrame,
    metrics: pd.DataFrame,
    value_col: str,
    config: DetectionConfig,
    context_cols: list[str],
) -> str:
    anomalies = detected[detected["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    start = series.index.min().strftime("%Y-%m-%d")
    end = series.index.max().strftime("%Y-%m-%d")
    anomaly_count = int(metric_value(metrics, "이상치 수"))
    anomaly_rate = metric_value(metrics, "이상치 비율")
    trend_strength = metric_value(metrics, "추세 강도")
    seasonality_strength = metric_value(metrics, "계절성 강도")
    lag1 = metric_value(metrics, "1시차 자기상관")
    cv = metric_value(metrics, "변동계수")

    if series.iloc[-1] > series.iloc[0]:
        direction = "상승"
    elif series.iloc[-1] < series.iloc[0]:
        direction = "하락"
    else:
        direction = "횡보"

    if anomaly_count == 0:
        anomaly_sentence = "현재 설정에서는 이상치가 탐지되지 않았습니다."
        review_sentence = "현재 임계값은 이 데이터에 대해 비교적 보수적으로 작동하고 있습니다."
    else:
        top = anomalies.iloc[0]
        anomaly_sentence = (
            f"총 {anomaly_count}개의 이상치가 탐지되었습니다(전체의 {anomaly_rate:.1%}). "
            f"가장 강한 이상 신호는 {top['timestamp'].strftime('%Y-%m-%d')}이며 "
            f"값은 {top['value']:.2f}, 유형은 {top['anomaly_type']}입니다."
        )
        review_sentence = (
            f"주요 검토 포인트: {top['recommended_action']} "
            f"탐지 근거: {top['reason']}."
        )

    structure_bits: list[str] = []
    if np.isfinite(trend_strength) and trend_strength >= 0.55:
        structure_bits.append(f"추세가 강함({trend_strength:.2f})")
    elif np.isfinite(trend_strength):
        structure_bits.append(f"추세가 보통/약함({trend_strength:.2f})")
    if np.isfinite(seasonality_strength) and seasonality_strength >= 0.45:
        structure_bits.append(f"계절성이 뚜렷함({seasonality_strength:.2f})")
    elif np.isfinite(seasonality_strength):
        structure_bits.append(f"계절성이 제한적임({seasonality_strength:.2f})")
    if np.isfinite(lag1):
        structure_bits.append(f"1시차 자기상관 {lag1:.2f}")
    if np.isfinite(cv):
        structure_bits.append(f"상대 변동성 {cv:.2f}")
    structure_sentence = "; ".join(structure_bits) if structure_bits else "데이터 길이가 짧아 구조 판단이 제한적입니다."

    context_sentence = (
        f"참고 변수로 {', '.join(context_cols)} 컬럼을 사용했습니다."
        if context_cols
        else "참고 변수를 선택하지 않아 대상 시계열만으로 판단했습니다."
    )

    return (
        f"**`{value_col}` 분석 브리핑**\n\n"
        f"- 분석 기간: {start} ~ {end}, 총 {len(series):,}개 관측치입니다. 전체 방향은 **{direction}**입니다.\n"
        f"- 시계열 구조: {structure_sentence}.\n"
        f"- 이상탐지 결과: {anomaly_sentence}\n"
        f"- 해석: {review_sentence}\n"
        f"- 참고 변수: {context_sentence}\n"
        f"- 현재 설정: {config.business_mode} 모드, 민감도 분위수 {config.sensitivity:.2f}, 이동 창 {config.window}."
    )


def generate_visual_insights(detected: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    anomalies = detected[detected["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    score_q90 = detected["anomaly_score"].quantile(0.90)
    score_q95 = detected["anomaly_score"].quantile(0.95)
    max_score = detected["anomaly_score"].max()
    dominant_component = (
        detected[["rolling_z", "iqr_score", "stl_score", "isolation_score"]]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    dominant_type = "none"
    if not anomalies.empty:
        dominant_type = anomalies["anomaly_type"].value_counts().idxmax()

    trend_strength = metric_value(metrics, "추세 강도")
    seasonality_strength = metric_value(metrics, "계절성 강도")
    anomaly_rate = metric_value(metrics, "이상치 비율")

    if np.isfinite(seasonality_strength) and seasonality_strength >= 0.6:
        structure_comment = "계절성이 강하므로 반복 패턴을 제거한 뒤에도 남는 이탈을 중요하게 봅니다."
    elif np.isfinite(trend_strength) and trend_strength >= 0.6:
        structure_comment = "추세가 뚜렷하므로 고정 평균이 아니라 이동 기준선과 비교합니다."
    else:
        structure_comment = "추세/계절 구조가 약해 국소 이상치 기준의 비중이 커집니다."

    if anomaly_rate <= 0.03:
        rate_comment = "이상치 비율이 낮아 선별적으로 탐지된 상태입니다."
    elif anomaly_rate <= 0.10:
        rate_comment = "이상치 비율이 중간 수준이므로 제거 전 개별 검토가 필요합니다."
    else:
        rate_comment = "이상치가 넓게 탐지되었습니다. 너무 많다면 민감도를 낮추는 것을 고려하세요."

    rows = [
        {
            "시각화 영역": "시계열 그래프",
            "확인할 내용": "빨간 표식은 국소 추세 또는 계절 패턴에서 벗어난 지점입니다.",
            "현재 해석": f"{len(anomalies)}개 지점이 표시되었고, 주요 이상 유형은 {dominant_type}입니다.",
        },
        {
            "시각화 영역": "이상 점수 추이",
            "확인할 내용": "점수가 높게 솟는 구간은 여러 탐지 기준이 강하게 반응한 시점입니다.",
            "현재 해석": f"최대 점수는 {max_score:.2f}, 90% 분위수는 {score_q90:.2f}, 95% 분위수는 {score_q95:.2f}입니다.",
        },
        {
            "시각화 영역": "점수 분포",
            "확인할 내용": "오른쪽 꼬리가 길수록 일부 관측치가 나머지보다 훨씬 특이하다는 뜻입니다.",
            "현재 해석": rate_comment,
        },
        {
            "시각화 영역": "탐지 기준별 점수",
            "확인할 내용": "rolling_z, IQR, STL, Isolation Forest 중 어떤 기준이 신호를 만들었는지 비교합니다.",
            "현재 해석": f"평균적으로 가장 강한 기준은 {dominant_component}입니다. {structure_comment}",
        },
    ]
    return pd.DataFrame(rows)


def data_quality_summary(meta: dict[str, float | int | str], metrics: pd.DataFrame, detected: pd.DataFrame) -> pd.DataFrame:
    missing = int(meta.get("missing_values", 0))
    duplicates = int(meta.get("duplicates", 0))
    points = int(metric_value(metrics, "관측치 수"))
    anomaly_rate = metric_value(metrics, "이상치 비율")
    adf_p = metric_value(metrics, "ADF p-value")

    if missing == 0 and duplicates == 0:
        quality_status = "양호"
        quality_note = "집계 전 대상 값 결측치와 중복 시점이 발견되지 않았습니다."
    elif missing <= max(1, points * 0.03) and duplicates <= max(1, points * 0.03):
        quality_status = "사용 가능"
        quality_note = "소량의 보간 또는 중복 집계만 필요했습니다."
    else:
        quality_status = "검토 필요"
        quality_note = "전처리 영향이 비교적 커서 이상치 판단을 신중히 검토해야 합니다."

    if np.isfinite(adf_p):
        stationarity_note = "비정상 시계열 가능성" if adf_p >= 0.05 else "정상 시계열 가능성"
    else:
        stationarity_note = "정상성 검정 판단 제한"

    return pd.DataFrame(
        [
            {"점검 항목": "데이터 품질", "상태": quality_status, "설명": quality_note},
            {"점검 항목": "시간 주기", "상태": str(meta.get("frequency", "unknown")), "설명": "자동 인식 또는 사용자가 선택한 시계열 주기입니다."},
            {"점검 항목": "정상성", "상태": stationarity_note, "설명": f"ADF p-value: {adf_p:.4f}" if np.isfinite(adf_p) else "ADF p-value를 계산할 수 없습니다."},
            {"점검 항목": "이상치 비율", "상태": f"{anomaly_rate:.1%}", "설명": "현재 이상치로 표시된 관측치의 비율입니다."},
        ]
    )


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


def apply_plot_theme(fig: go.Figure, height: int) -> go.Figure:
    axis_style = dict(
        showgrid=True,
        gridcolor=THEME["grid"],
        zeroline=False,
        color=THEME["ink"],
        tickfont=dict(color=THEME["ink"], size=12),
        title_font=dict(color=THEME["ink"], size=14),
        linecolor=THEME["grid"],
    )
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        font=dict(color=THEME["ink"], family="Noto Sans KR, Segoe UI, sans-serif", size=13),
        title_font=dict(color=THEME["ink"], size=18),
        margin=dict(l=10, r=10, t=38, b=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=THEME["ink"], size=12),
        ),
        hoverlabel=dict(
            bgcolor="#fbfbf8",
            bordercolor=THEME["coral"],
            font=dict(color=THEME["ink"], family="Noto Sans KR, Segoe UI, sans-serif", size=13),
        ),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def enforce_plot_text_color(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(color=THEME["ink"], family="Noto Sans KR, Segoe UI, sans-serif", size=13),
        title_font=dict(color=THEME["ink"], size=18),
        legend_font=dict(color=THEME["ink"], size=12),
        hoverlabel=dict(
            bgcolor="#fbfbf8",
            bordercolor=THEME["coral"],
            font=dict(color=THEME["ink"], family="Noto Sans KR, Segoe UI, sans-serif", size=13),
        ),
    )
    fig.update_xaxes(
        color=THEME["ink"],
        tickfont=dict(color=THEME["ink"], size=12),
        title_font=dict(color=THEME["ink"], size=14),
    )
    fig.update_yaxes(
        color=THEME["ink"],
        tickfont=dict(color=THEME["ink"], size=12),
        title_font=dict(color=THEME["ink"], size=14),
    )
    return fig


def line_chart(detected: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=detected["timestamp"],
            y=detected["value"],
            mode="lines",
            name="시계열",
            line=dict(color=THEME["line"], width=2.4),
        )
    )
    anomalies = detected[detected["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["value"],
            mode="markers",
            name="이상치",
            marker=dict(color=THEME["coral_dark"], size=11, symbol="x", line=dict(width=2)),
            text=anomalies["anomaly_type"],
            hovertemplate="%{x}<br>값=%{y}<br>%{text}<extra></extra>",
        )
    )
    apply_plot_theme(fig, 460)
    fig.update_layout(hovermode="x unified", title="시계열 흐름과 이상치")
    fig.update_xaxes(title="시간")
    fig.update_yaxes(title="값")
    return enforce_plot_text_color(fig)


def score_chart(detected: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12)
    fig.add_trace(
        go.Scatter(
            x=detected["timestamp"],
            y=detected["anomaly_score"],
            mode="lines",
            name="종합 이상 점수",
            line=dict(color=THEME["line"], width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=detected["anomaly_score"],
            nbinsx=30,
            marker_color=THEME["mint_strong"],
            name="점수 분포",
        ),
        row=2,
        col=1,
    )
    apply_plot_theme(fig, 520)
    fig.update_layout(title="종합 이상 점수와 분포")
    fig.update_yaxes(title="점수", row=1, col=1)
    fig.update_yaxes(title="개수", row=2, col=1)
    fig.update_xaxes(title="시간", row=1, col=1)
    fig.update_xaxes(title="점수", row=2, col=1)
    return enforce_plot_text_color(fig)


def component_chart(detected: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in [
        ("rolling_z", "#3f7f77"),
        ("iqr_score", "#7b4cc2"),
        ("stl_score", THEME["coral_dark"]),
        ("isolation_score", "#6f7689"),
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
    apply_plot_theme(fig, 430)
    fig.update_layout(hovermode="x unified", title="탐지 기준별 점수")
    fig.update_xaxes(title="시간")
    fig.update_yaxes(title="기준별 점수")
    return enforce_plot_text_color(fig)


def anomaly_type_chart(detected: pd.DataFrame) -> go.Figure:
    anomalies = detected[detected["is_anomaly"]]
    counts = anomalies["anomaly_type"].value_counts().reset_index()
    counts.columns = ["유형", "개수"]
    fig = go.Figure(
        go.Bar(
            x=counts["유형"],
            y=counts["개수"],
            marker_color=[THEME["coral"], THEME["mint_strong"], "#f2c078", "#8ab6a8", "#6f7689"][: len(counts)],
        )
    )
    apply_plot_theme(fig, 330)
    fig.update_layout(title="이상치 유형 분포")
    fig.update_xaxes(title="이상치 유형")
    fig.update_yaxes(title="개수")
    return enforce_plot_text_color(fig)


def stationarity_diagnostics(series: pd.Series) -> pd.DataFrame:
    values = series.dropna()
    diff_values = values.diff().dropna()

    def adf_summary(label: str, sample: pd.Series) -> dict[str, str | float]:
        if len(sample) < 12 or sample.nunique() <= 1:
            return {"구분": label, "ADF p-value": np.nan, "판정": "계산 제한", "해석": "관측치가 부족하거나 값 변화가 거의 없습니다."}
        try:
            p_value = float(adfuller(sample, autolag="AIC")[1])
        except Exception:
            p_value = np.nan
        if np.isfinite(p_value) and p_value < 0.05:
            verdict = "정상성 가능"
            note = "평균과 분산 구조가 비교적 안정적이라고 볼 수 있습니다."
        elif np.isfinite(p_value):
            verdict = "비정상 가능"
            note = "추세나 계절성을 제거하거나 차분 후 분석하는 것이 유리할 수 있습니다."
        else:
            verdict = "계산 제한"
            note = "ADF 검정을 계산할 수 없습니다."
        return {"구분": label, "ADF p-value": p_value, "판정": verdict, "해석": note}

    return pd.DataFrame([adf_summary("원시 시계열", values), adf_summary("1차 차분", diff_values)])


def acf_pacf_chart(series: pd.Series) -> go.Figure:
    values = series.dropna()
    max_lag = max(1, min(30, len(values) // 3))
    acf_values = acf(values, nlags=max_lag, fft=False)
    try:
        pacf_values = pacf(values, nlags=max_lag, method="ywm")
    except Exception:
        pacf_values = np.full(max_lag + 1, np.nan)
    lags = list(range(max_lag + 1))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF 자기상관", "PACF 부분자기상관"))
    fig.add_trace(go.Bar(x=lags, y=acf_values, marker_color=THEME["mint_strong"], name="ACF"), row=1, col=1)
    fig.add_trace(go.Bar(x=lags, y=pacf_values, marker_color=THEME["coral"], name="PACF"), row=1, col=2)
    conf = 1.96 / np.sqrt(max(len(values), 1))
    for col in [1, 2]:
        fig.add_hline(y=conf, line_dash="dot", line_color=THEME["muted"], row=1, col=col)
        fig.add_hline(y=-conf, line_dash="dot", line_color=THEME["muted"], row=1, col=col)
    apply_plot_theme(fig, 390)
    fig.update_layout(title="ACF/PACF 진단")
    fig.update_xaxes(title="시차")
    fig.update_yaxes(title="상관")
    return enforce_plot_text_color(fig)


def stl_decomposition_chart(series: pd.Series, freq: str) -> go.Figure:
    period = guess_period(freq, len(series))
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    values = series.dropna()
    if len(values) >= max(24, period * 2):
        try:
            fit = STL(values, period=period, robust=True).fit()
            panels = [
                ("원시값", values, THEME["line"]),
                ("추세", fit.trend, "#3f7f77"),
                ("계절", fit.seasonal, THEME["coral_dark"]),
                ("잔차", fit.resid, "#6f7689"),
            ]
        except Exception:
            panels = [("원시값", values, THEME["line"])]
    else:
        panels = [("원시값", values, THEME["line"])]

    for idx, (name, sample, color) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=sample.index, y=sample.values, mode="lines", name=name, line=dict(color=color, width=1.8)), row=idx, col=1)
    apply_plot_theme(fig, 620)
    fig.update_layout(title=f"STL 분해 진단 (period={period})")
    fig.update_xaxes(title="시간", row=len(panels), col=1)
    fig.update_yaxes(title="값")
    return enforce_plot_text_color(fig)


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
    briefing = generate_analysis_briefing(
        pd.Series(detected["value"].values, index=pd.to_datetime(detected["timestamp"])),
        detected,
        metrics,
        value_col,
        config,
        context_cols,
    ).replace("**", "")
    lines = [
        "# 시계열 이상탐지 분석 리포트",
        "",
        f"- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 시간 컬럼: `{time_col}`",
        f"- 분석 대상 컬럼: `{value_col}`",
        f"- 참고 변수: `{', '.join(context_cols) if context_cols else '없음'}`",
        f"- 분석 모드: `{config.business_mode}`",
        f"- 시간 주기: `{config.freq}`",
        f"- 이동 창: `{config.window}`",
        f"- 민감도 분위수: `{config.sensitivity:.2f}`",
        "",
        "## 분석 브리핑",
        briefing,
        "",
        "## 주요 지표",
    ]
    for _, row in metrics.iterrows():
        value = row["값"]
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"- {row['지표']}: {value} ({row['의미']})")
    lines.extend(["", "## 주요 이상치"])
    if top.empty:
        lines.append("- 현재 설정에서 탐지된 이상치가 없습니다.")
    else:
        for _, row in top.iterrows():
            lines.append(
                f"- {row['timestamp']}: 값={row['value']:.4f}, 점수={row['anomaly_score']:.4f}, "
                f"유형={row['anomaly_type']}, 근거={row['reason']}, 권장 조치={row['recommended_action']}"
            )
    return "\n".join(lines)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700;800&display=swap');

        :root {
            --ink: #242331;
            --muted: #6f7f7b;
            --paper: #fbfbf8;
            --panel: #ffffff;
            --mint: #dfeeea;
            --mint-strong: #b7dfa8;
            --coral: #e78c73;
            --coral-dark: #c96953;
            --line: #2c2b3b;
            --grid: #d8e3df;
        }

        html, body, [class*="css"] {
            font-family: 'Noto Sans KR', 'Segoe UI', sans-serif;
        }

        .stApp {
            background:
                linear-gradient(175deg, rgba(183, 205, 199, .55) 0 18%, transparent 18%),
                linear-gradient(5deg, transparent 0 83%, rgba(183, 205, 199, .48) 83%),
                var(--mint);
            color: var(--ink);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.3rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background: var(--ink);
            border-right: 0;
        }

        [data-testid="stSidebar"] * {
            color: #f8faf8 !important;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p {
            font-weight: 700;
        }

        [data-testid="stSidebar"] section {
            background: transparent;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] input {
            background: #343342 !important;
            border-color: rgba(255,255,255,.14) !important;
        }

        .dashboard-shell {
            background: rgba(251, 251, 248, .96);
            border: 1px solid rgba(36,35,49,.08);
            box-shadow: 0 28px 70px rgba(36,35,49,.16);
            padding: 28px 30px 34px;
            margin: 8px 0 22px;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            padding: 8px 0 22px;
            border-bottom: 1px solid rgba(36,35,49,.08);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 270px;
        }

        .brand-mark {
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--ink) 0 52%, var(--coral) 52%);
            color: #fff;
            font-weight: 900;
            letter-spacing: 0;
        }

        .brand-title {
            font-size: 1.05rem;
            font-weight: 900;
            letter-spacing: .04em;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: .86rem;
            margin-top: 1px;
        }

        .top-pill {
            flex: 1;
            max-width: 420px;
            background: #edf5f3;
            border-radius: 999px;
            padding: 8px 16px;
            color: var(--muted);
            font-size: .9rem;
            text-align: center;
        }

        .student-badge {
            background: var(--coral);
            color: var(--ink);
            border-radius: 999px;
            padding: 9px 14px;
            font-weight: 900;
            white-space: nowrap;
        }

        .page-heading {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 20px;
            margin: 22px 0 18px;
        }

        .page-heading h1 {
            margin: 0;
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.4rem);
            letter-spacing: 0;
            line-height: 1;
            font-weight: 900;
        }

        .page-heading p {
            margin: 8px 0 0;
            color: var(--muted);
            font-weight: 600;
        }

        .kpi-card {
            background: var(--panel);
            border: 1px solid rgba(36,35,49,.08);
            border-radius: 8px;
            min-height: 138px;
            box-shadow: 0 10px 24px rgba(36,35,49,.07);
            overflow: hidden;
        }

        .kpi-head {
            background: var(--ink);
            color: #fff;
            padding: 11px 16px;
            font-weight: 900;
            letter-spacing: .02em;
            font-size: .92rem;
        }

        .kpi-body {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 16px 16px 14px;
        }

        .kpi-value {
            color: var(--ink);
            font-weight: 900;
            font-size: clamp(1.55rem, 2.5vw, 2.25rem);
            line-height: 1;
        }

        .kpi-caption {
            color: var(--muted);
            font-weight: 800;
            font-size: .78rem;
            margin: 0 16px 14px;
        }

        .kpi-icon {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: var(--coral);
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 900;
            flex: 0 0 auto;
        }

        .kpi-icon.mint { background: var(--mint-strong); }
        .kpi-icon.dark { background: var(--ink); color: #fff; }

        div[data-testid="stTabs"] button {
            color: var(--ink);
            font-weight: 900;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--coral-dark);
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--coral);
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        [data-testid="stExpander"],
        .stDownloadButton button {
            border-radius: 8px !important;
        }

        [data-testid="stExpander"] {
            background: rgba(255,255,255,.5) !important;
            border: 1px solid rgba(36,35,49,.07) !important;
            box-shadow: 0 8px 22px rgba(36,35,49,.04);
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            background: #eef7f4 !important;
            color: var(--ink) !important;
            border-radius: 8px 8px 0 0 !important;
            font-weight: 900 !important;
        }

        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] summary svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
            stroke: var(--ink) !important;
        }

        .stDownloadButton button {
            background: var(--coral) !important;
            color: var(--ink) !important;
            border: 0 !important;
            font-weight: 900 !important;
            box-shadow: 0 8px 18px rgba(201,105,83,.18);
        }

        .stDownloadButton button:hover {
            background: var(--coral-dark) !important;
            color: #fff !important;
            border: 0 !important;
        }

        .stCaption, .stMarkdown p, .stWrite {
            color: var(--ink);
        }

        [data-testid="stPlotlyChart"] svg text {
            fill: var(--ink) !important;
            color: var(--ink) !important;
        }

        [data-testid="stPlotlyChart"] .legend text,
        [data-testid="stPlotlyChart"] .gtitle,
        [data-testid="stPlotlyChart"] .xtitle,
        [data-testid="stPlotlyChart"] .ytitle {
            fill: var(--ink) !important;
        }

        [data-testid="stPlotlyChart"] .hoverlayer .hovertext path {
            fill: var(--paper) !important;
            stroke: var(--coral) !important;
        }

        [data-testid="stPlotlyChart"] .hoverlayer .hovertext text {
            fill: var(--ink) !important;
        }

        @media (max-width: 820px) {
            .dashboard-shell { padding: 18px 14px; }
            .topbar, .page-heading { align-items: flex-start; flex-direction: column; }
            .top-pill { max-width: none; width: 100%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header(value_col: str, context_cols: list[str]) -> None:
    context_text = f"{len(context_cols)}개 참고 변수 사용" if context_cols else "대상 시계열 단독 분석"
    st.markdown(
        f"""
        <div class="dashboard-shell">
            <div class="topbar">
                <div class="brand">
                    <div class="brand-mark">TS</div>
                    <div>
                        <div class="brand-title">ANOMALY INTELLIGENCE</div>
                        <div class="brand-subtitle">시계열분석 프로젝트 2</div>
                    </div>
                </div>
                <div class="top-pill">대상 컬럼: {value_col} · {context_text}</div>
                <div class="student-badge">C321083 김태환</div>
            </div>
            <div class="page-heading">
                <div>
                    <h1>이상탐지 대시보드</h1>
                    <p>CSV 업로드, 자동 탐지, 강의 기반 진단, 설명형 검토 리포트를 한 화면에서 확인합니다.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, caption: str, icon: str, tone: str = "dark") -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-head">{title}</div>
            <div class="kpi-body">
                <div class="kpi-value">{value}</div>
                <div class="kpi-icon {tone}">{icon}</div>
            </div>
            <div class="kpi-caption">{caption}</div>
        </div>
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

    with st.sidebar:
        st.header("입력 데이터")
        uploaded_file = st.file_uploader("CSV 파일", type=["csv"])
        df = read_csv(uploaded_file)

        dt_cols = candidate_datetime_columns(df)
        num_cols = candidate_numeric_columns(df)
        if not dt_cols:
            st.error("날짜/시간 컬럼을 찾지 못했습니다.")
            return
        if not num_cols:
            st.error("수치형 컬럼을 찾지 못했습니다.")
            return

        time_col = st.selectbox("시간 컬럼", dt_cols, index=0)
        value_default = 0 if num_cols[0] != time_col else min(1, len(num_cols) - 1)
        value_col = st.selectbox("분석 대상 값 컬럼", num_cols, index=value_default)
        context_options = [col for col in num_cols if col not in {time_col, value_col}]
        context_cols = st.multiselect("원인 힌트용 참고 변수", context_options, default=context_options[:2])

        st.header("탐지 설정")
        freq = st.selectbox("시간 주기", ["auto", "D", "W", "M", "H"], index=0)
        window = st.slider("이동 창 크기", min_value=5, max_value=90, value=21, step=2)
        sensitivity_label = st.select_slider(
            "탐지 민감도",
            options=["낮음", "보통", "높음", "매우 높음"],
            value="높음",
        )
        sensitivity_map = {"낮음": 0.985, "보통": 0.965, "높음": 0.94, "매우 높음": 0.90}
        use_stl = st.toggle("STL 잔차 점수 사용", value=True)
        use_iforest = st.toggle("다변량 Isolation Forest 사용", value=True)
        business_mode = st.selectbox(
            "검토 모드",
            ["수요/매출", "운영/센서", "금융/리스크", "일반"],
            index=0,
        )

    try:
        series, meta = prepare_series(df, time_col, value_col, freq)
        context = prepare_context_frame(df, time_col, context_cols, series.index)
    except Exception as exc:
        st.error(f"시계열 전처리에 실패했습니다: {exc}")
        return

    if len(series) < 12:
        st.error("분석에는 최소 12개 이상의 유효 관측치를 권장합니다.")
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
    briefing = generate_analysis_briefing(series, detected, metrics, value_col, config, context_cols)
    visual_insights = generate_visual_insights(detected, metrics)
    quality_summary = data_quality_summary(meta, metrics, detected)
    anomaly_count = int(detected["is_anomaly"].sum())
    anomaly_rate = anomaly_count / len(detected)
    latest = detected[detected["is_anomaly"]].tail(1)
    latest_text = "none" if latest.empty else str(latest.iloc[0]["timestamp"])

    render_top_header(value_col, context_cols)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("관측치", f"{len(series):,}", "분석에 사용된 시점", "∑", "dark")
    with k2:
        render_kpi_card("이상치", f"{anomaly_count:,}", "탐지된 검토 후보", "!", "coral")
    with k3:
        render_kpi_card("이상치 비율", f"{anomaly_rate:.1%}", "전체 대비 이상치", "%", "mint")
    with k4:
        render_kpi_card("참고 변수", f"{len(context_cols):,}", "다변량 분석 입력", "+", "dark")
    with k5:
        render_kpi_card("최근 이상치", latest_text[:16], "가장 마지막 탐지 시점", "⌁", "coral")

    tabs = st.tabs(
        [
            "대시보드",
            "이상치 검토",
            "평가지표",
            "시계열 진단",
            "데이터 미리보기",
        ]
    )
    with tabs[0]:
        st.subheader("분석 브리핑")
        st.markdown(briefing)
        with st.expander("시각화 해석 가이드", expanded=True):
            stretch_dataframe(visual_insights, hide_index=True)
        with st.expander("데이터 품질 점검", expanded=False):
            stretch_dataframe(quality_summary, hide_index=True)
        st.divider()
        stretch_plotly_chart(line_chart(detected))
        c1, c2 = st.columns([1, 1])
        with c1:
            stretch_plotly_chart(score_chart(detected))
            st.caption("점수 추이와 분포는 각 탐지 지점이 전체 시계열에서 얼마나 드문지 보여줍니다.")
        with c2:
            stretch_plotly_chart(component_chart(detected))
            st.caption("기준별 점수는 국소 이탈, IQR, 계절성 잔차, 다변량 고립 기준 중 어떤 신호가 강했는지 보여줍니다.")

    with tabs[1]:
        c1, c2 = st.columns([1.15, 0.85])
        anomalies = detected[detected["is_anomaly"]].sort_values("anomaly_score", ascending=False)
        with c1:
            review_cols = {
                "timestamp": "시점",
                "value": "값",
                "anomaly_score": "이상 점수",
                "severity": "심각도",
                "method_votes": "동의 기준 수",
                "anomaly_type": "이상 유형",
                "reason": "탐지 근거",
                "context_note": "참고 변수 해석",
                "recommended_action": "권장 검토 조치",
            }
            stretch_dataframe(anomalies[list(review_cols)].rename(columns=review_cols), hide_index=True)
        with c2:
            stretch_plotly_chart(anomaly_type_chart(detected))
            st.caption("각 이상치는 유형, 탐지 근거, 권장 검토 조치와 함께 정리됩니다.")

        st.download_button(
            "이상탐지 결과 CSV 다운로드",
            data=dataframe_to_csv_bytes(detected),
            file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
        report = make_report(detected, metrics, time_col, value_col, config, context_cols)
        st.download_button(
            "분석 리포트 다운로드",
            data=report.encode("utf-8-sig"),
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    with tabs[2]:
        stretch_dataframe(metrics, hide_index=True)
        st.caption(
            "종합 이상 점수는 rolling z-score, IQR 거리, STL 잔차 점수, 다변량 Isolation Forest 점수를 결합해 계산합니다."
        )

    with tabs[3]:
        st.subheader("강의 기반 시계열 진단")
        st.write("정상성, 차분, 자기상관, 부분자기상관, STL 분해를 함께 확인해 이상탐지 결과의 배경 구조를 판단합니다.")
        d1, d2 = st.columns([0.9, 1.1])
        with d1:
            st.markdown("#### 정상성 및 차분 비교")
            stretch_dataframe(stationarity_diagnostics(series), hide_index=True)
            st.caption("ADF p-value가 0.05보다 작으면 정상 시계열 가능성이 높다고 해석합니다.")
        with d2:
            stretch_plotly_chart(acf_pacf_chart(series))
            st.caption("ACF/PACF는 시차별 의존성을 보여주며, AR/MA 구조 또는 계절 패턴 판단에 활용됩니다.")
        stretch_plotly_chart(stl_decomposition_chart(series, config.freq))
        st.caption("STL 분해는 원시 시계열을 추세, 계절, 잔차로 나누어 계절성으로 설명되지 않는 이탈을 확인합니다.")

    with tabs[4]:
        st.write("업로드 파일, 분석 대상 컬럼, 참고 변수, 탐지 설정을 바꾸면 분석 결과가 자동으로 다시 계산됩니다.")
        stretch_dataframe(df.head(200))


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)
        raise SystemExit(0)
    main()
