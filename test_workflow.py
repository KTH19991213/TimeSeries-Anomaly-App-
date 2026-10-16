import json
import unittest
from dataclasses import replace

import numpy as np
import app
from benchmark import synthetic_case
from workflow import make_manifest, restore_reviews, level_shift_signal


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = app.DetectionConfig("D", 21, .94, True, True, "일반")
        self.manifest = make_manifest(b"date,value\n", "sample.csv", self.config,
                                      {"aggregation": "mean"}, "value", [], {"domain": "일반"})

    def test_manifest_identity_changes_with_file_and_conditions(self):
        duplicate = make_manifest(b"date,value\n", "sample.csv", self.config, {"aggregation": "mean"}, "value", [], {"domain": "일반"})
        self.assertEqual(self.manifest, duplicate)
        changed = make_manifest(b"date,value\n1,2", "sample.csv", self.config, {"aggregation": "mean"}, "value", [], {"domain": "일반"})
        self.assertNotEqual(self.manifest["analysis_id"], changed["analysis_id"])
        changed_config = make_manifest(b"date,value\n", "sample.csv", replace(self.config, period=14), {"aggregation": "mean"}, "value", [], {"domain": "일반"})
        self.assertNotEqual(self.manifest["analysis_id"], changed_config["analysis_id"])

    def test_review_roundtrip_and_mismatched_rejection(self):
        package = {"schema_version": 1, "analysis_id": self.manifest["analysis_id"],
                   "reviews": {"2026-01-01T00:00:00": {"decision": "실제 이상", "note": "검토 메모"}}}
        self.assertEqual(restore_reviews(json.loads(json.dumps(package)), self.manifest), package["reviews"])
        with self.assertRaises(ValueError):
            restore_reviews({**package, "analysis_id": "another"}, self.manifest)
        with self.assertRaises(ValueError):
            restore_reviews([], self.manifest)

    def test_shift_labels_align_and_improve_unseen_seeds(self):
        for seed in (127, 2026):
            series, truth = synthetic_case(seed, "지속 수준 변화")
            original = app.detect_anomalies(series, replace(self.config, use_level_shift=False))
            improved = app.detect_anomalies(series, self.config)
            self.assertTrue(improved.level_shift_score.notna().all())
            self.assertGreater(np.sum(truth & improved.is_anomaly.to_numpy()), np.sum(truth & original.is_anomaly.to_numpy()))
            self.assertFalse(improved.loc[improved.is_anomaly, "severity"].eq("정상").any())

    def test_shift_not_triggered_on_heldout_normal_patterns(self):
        for seed in (127, 2026):
            series, _ = synthetic_case(seed, "정상 패턴")
            self.assertFalse(level_shift_signal(series, 7, 21).ge(3.5).any())


if __name__ == "__main__":
    unittest.main()
