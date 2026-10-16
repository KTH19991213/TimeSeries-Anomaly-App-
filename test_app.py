import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import app
from benchmark import evaluate_labels, run_benchmark
from intelligence import infer_profile


class DataUnderstandingTests(unittest.TestCase):
    def test_unknown_data_does_not_invent_company_or_domain(self):
        result = infer_profile(pd.DataFrame(columns=["date", "x", "y"]))
        self.assertEqual(result.domain, "일반·미확인")
        self.assertEqual(result.company, "미확인")

    def test_industry_evidence_and_user_override(self):
        frame = pd.DataFrame(columns=["date", "sales", "inventory"])
        self.assertEqual(infer_profile(frame).domain, "유통·수요")
        result = infer_profile(frame, company="사용자 기업", override="제조·센서")
        self.assertEqual((result.domain, result.company, result.certainty), ("제조·센서", "사용자 기업", "사용자 지정"))

    def test_ambiguous_and_substring_names_abstain(self):
        self.assertEqual(infer_profile(pd.DataFrame(columns=["sales", "sensor"])).domain, "일반·미확인")
        self.assertEqual(infer_profile(pd.DataFrame(columns=["disclose", "payload"])).domain, "일반·미확인")


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=59), "value": np.ones(59), "sensor": np.arange(59)})

    def test_monthly_sum_keeps_all_days(self):
        series, meta = app.prepare_series(self.frame, "date", "value", "ME", "sum")
        self.assertEqual(list(series), [31, 28])
        self.assertEqual(meta["aggregation"], "sum")

    def test_monthly_context_uses_same_bins(self):
        series, _ = app.prepare_series(self.frame, "date", "value", "ME")
        context = app.prepare_context_frame(self.frame, "date", ["sensor"], series.index, "ME")
        self.assertEqual(list(context.sensor), [15, 44.5])

    def test_short_and_invalid_series_have_clear_outcomes(self):
        series, _ = app.prepare_series(self.frame.head(2), "date", "value", "auto")
        self.assertEqual(len(series), 2)
        with self.assertRaisesRegex(ValueError, "유효한"):
            app.prepare_series(pd.DataFrame({"date": ["bad"], "value": [1]}), "date", "value", "auto")

    def test_utf8_cp949_and_delimiter(self):
        for encoding in ("utf-8-sig", "cp949"):
            source = io.BytesIO("날짜;매출\n2026-01-01;12\n2026-01-02;13\n".encode(encoding))
            self.assertEqual(app.read_csv(source)["매출"].sum(), 25)

    def test_gap_counts_and_duplicate_aggregation(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-01", "2026-01-03"], "value": [1, 3, 6]})
        series, meta = app.prepare_series(frame, "date", "value", "D", "sum")
        self.assertEqual(list(series), [4, 5, 6])
        self.assertEqual((meta["duplicates"], meta["inserted_missing"]), (1, 1))

    def test_excessive_expansion_rejected(self):
        frame = pd.DataFrame({"date": ["1900-01-01", "2026-01-01"], "value": [1, 2]})
        with self.assertRaisesRegex(ValueError, "20만"):
            app.prepare_series(frame, "date", "value", "h")


class DetectionTests(unittest.TestCase):
    def test_near_zero_mad_does_not_explode(self):
        values = pd.Series([1e-12, -1e-12] * 20 + [100.0])
        scores = app.robust_z_score(values)
        self.assertTrue(np.isfinite(scores).all())
        self.assertLess(scores.abs().max(), 10)

    def test_short_seasonal_tourism_scores_are_finite(self):
        from pathlib import Path
        frame = pd.read_csv(Path(__file__).with_name("australian_tourism_monthly.csv"))
        series, _ = app.prepare_series(frame, "Month", "Total", "auto")
        result = app.detect_anomalies(series, app.DetectionConfig("MS", 18, .94, True, True, "일반"))
        self.assertLess(result.anomaly_score.max(), 100)

    def test_vote_trigger_cannot_have_normal_severity(self):
        series = pd.Series(np.arange(30.), index=pd.date_range("2026-01-01", periods=30))
        scores = pd.Series(2.5, index=series.index)
        with patch.object(app, "rolling_z_score", return_value=scores), patch.object(app, "iqr_score", return_value=scores * .4), patch.object(app, "stl_score", return_value=scores), patch.object(app, "isolation_forest_score", return_value=scores * 0):
            result = app.detect_anomalies(series, app.DetectionConfig("D", 9, .94, True, True, "일반"))
        self.assertTrue(result.is_anomaly.all())
        self.assertFalse(result.severity.eq("정상").any())

    def test_constant_series_and_disabled_methods(self):
        series = pd.Series(np.ones(30), index=pd.date_range("2026-01-01", periods=30))
        result = app.detect_anomalies(series, app.DetectionConfig("D", 9, .94, False, False, "일반"))
        self.assertFalse(result.is_anomaly.any())
        self.assertTrue(result.active_methods.eq(2).all())

    def test_label_metrics_and_normal_case(self):
        result = evaluate_labels([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((result["Precision"], result["Recall"], result["F1"]), (.5, .5, .5))
        self.assertTrue(np.isnan(evaluate_labels([0, 0], [0, 1])["F1"]))


if __name__ == "__main__":
    unittest.main()
