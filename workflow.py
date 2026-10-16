"""Retrospective shift detection and portable, dataset-scoped analysis records."""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import pandas as pd


def level_shift_signal(series: pd.Series, period: int, window: int) -> pd.Series:
    """Remove a robust linear slope and phase medians, then segment median changes.

    Adjacent windows propose boundaries. Sustained segments are flagged only when
    their median differs from the full-series median by at least 3.5 noise scales.
    This uses both sides of a boundary and is explicitly retrospective.
    """
    n = len(series)
    output = np.zeros(n)
    span = max(4, min(window // 2, period, n // 6))
    if n < max(24, 4 * span):
        return pd.Series(output, index=series.index)
    values = series.to_numpy(dtype=float)
    lag = min(period, n // 4)
    slope = np.median((values[lag:] - values[:-lag]) / lag)
    detrended = values - slope * np.arange(n)
    phases = np.arange(n) % period
    seasonal = np.array([np.median(detrended[phases == phase]) for phase in np.unique(phases)])
    residual = detrended - seasonal[phases]
    center = np.median(residual)
    noise = 1.4826 * np.median(np.abs(residual - center))
    if noise <= np.finfo(float).eps * max(1., np.max(np.abs(values))) * 256:
        noise = max(np.std(residual), 1e-9)
    boundary_scores = np.zeros(n)
    for i in range(span, n - span + 1):
        boundary_scores[i] = abs(np.median(residual[i:i + span]) - np.median(residual[i - span:i])) / noise
    candidates = np.flatnonzero(boundary_scores >= 3.5)
    groups = np.split(candidates, np.where(np.diff(candidates) > 1)[0] + 1)
    boundaries = [int(group[np.argmax(boundary_scores[group])]) for group in groups if len(group)]
    # Avoid one-point segments between adjacent change estimates.
    selected = []
    for point in sorted(boundaries, key=lambda i: boundary_scores[i], reverse=True):
        if all(abs(point - other) >= span for other in selected):
            selected.append(point)
    edges = [0, *sorted(selected), n]
    for start, end in zip(edges, edges[1:]):
        strength = abs(np.median(residual[start:end]) - center) / noise
        if end - start >= span and strength >= 3.5:
            output[start:end] = strength
    return pd.Series(output, index=series.index)


def make_manifest(raw: bytes, filename: str, config, preprocessing: dict,
                  target: str, context_cols: list[str], profile: dict) -> dict:
    payload = {"schema_version": 1, "algorithm_version": "portfolio-2",
               "dataset_sha256": hashlib.sha256(raw).hexdigest(), "filename": filename,
               "detection": asdict(config), "preprocessing": preprocessing,
               "target": target, "context_columns": context_cols, "profile": profile,
               "environment": {"python": platform.python_version(), **{name: version(name) for name in
                   ("streamlit", "pandas", "numpy", "scikit-learn", "statsmodels", "plotly")}}}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    payload["analysis_id"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def restore_reviews(data: dict, manifest: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("검토 기록은 JSON 객체여야 합니다.")
    if data.get("schema_version") != 1 or data.get("analysis_id") != manifest["analysis_id"]:
        raise ValueError("데이터 또는 분석 조건이 현재 분석과 다릅니다. 같은 파일·컬럼·설정에서 불러오세요.")
    records = data.get("reviews")
    if not isinstance(records, dict):
        raise ValueError("검토 기록 형식이 올바르지 않습니다.")
    cleaned = {}
    for timestamp, record in records.items():
        if not isinstance(record, dict) or record.get("decision") not in {"실제 이상", "정상·오탐", "판단 보류"}:
            raise ValueError("검토 판단 형식이 올바르지 않습니다.")
        pd.Timestamp(timestamp)
        if not isinstance(record.get("note", ""), str):
            raise ValueError("검토 메모 형식이 올바르지 않습니다.")
        cleaned[timestamp] = {"decision": record["decision"], "note": record.get("note", "")[:4000]}
    return cleaned
