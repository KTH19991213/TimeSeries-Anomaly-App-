"""Reproducible, point-level synthetic checks; not a real-world accuracy claim."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace


def synthetic_case(seed: int, scenario: str) -> tuple[pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 180
    values = 70 + np.linspace(0, 15, n) + 8 * np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 2, n)
    labels = np.zeros(n, dtype=bool)
    if scenario == "단일 급등·급락":
        points = np.array([22, 48, 79, 121, 153])
        values[points] += np.array([28, -24, 35, -30, 26])
        labels[points] = True
    elif scenario == "지속 수준 변화":
        values[90:105] += 24
        labels[90:105] = True
    elif scenario != "정상 패턴":
        raise ValueError("알 수 없는 시나리오")
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=n, freq="D")), labels


def evaluate_labels(truth: np.ndarray, predicted: np.ndarray) -> dict:
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("정답과 예측은 같은 길이의 1차원 배열이어야 합니다.")
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if truth.any() else np.nan
    return {"Precision": precision, "Recall": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn}


def run_benchmark(detector, baseline, config) -> pd.DataFrame:
    rows = []
    for scenario in ("정상 패턴", "단일 급등·급락", "지속 수준 변화"):
        for seed in (11, 42, 83):
            series, truth = synthetic_case(seed, scenario)
            ensemble = detector(series, config)
            original = detector(series, replace(config, use_level_shift=False))
            for method, predicted in (("앙상블 + 수준 변화", ensemble.is_anomaly.to_numpy()),
                                      ("기존 앙상블", original.is_anomaly.to_numpy()),
                                      ("이동 창 z-score 단독", baseline(series, config.window).abs().ge(2.5).to_numpy())):
                rows.append({"시나리오": scenario, "seed": seed, "방법": method,
                             **evaluate_labels(truth, predicted)})
    return pd.DataFrame(rows)
