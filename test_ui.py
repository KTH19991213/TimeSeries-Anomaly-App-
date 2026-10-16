"""Local Streamlit regression tests: no cloud account or API key required."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class DashboardFlowTests(unittest.TestCase):
    def test_profile_questions_and_reproducible_evaluation(self):
        at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=40).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.tabs[0].label, "데이터 이해")
        at.text_input[0].set_value("포트폴리오 예시 기관")
        at.text_area[0].set_value("장비 온도와 압력을 분석하는 합성 예시")
        next(s for s in at.selectbox if s.label == "분야 확인·수정").set_value("제조·센서").run()
        self.assertEqual(len(at.exception), 0)
        paragraphs = " ".join(x.value for x in at.markdown)
        self.assertIn("제조·센서", paragraphs)
        self.assertIn("포트폴리오 예시 기관", paragraphs)
        self.assertIn("정비", paragraphs)
        for question in at.radio[0].options:
            at.radio[0].set_value(question).run()
            self.assertEqual(len(at.exception), 0)
        next(b for b in at.button if b.label == "재현 가능한 성능 평가 실행").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.session_state["synthetic_evaluation"]), 27)
        at.run()
        self.assertEqual(len(at.session_state["synthetic_evaluation"]), 27)

    def test_review_save_isolation_and_custom_period(self):
        at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=40).run()
        self.assertEqual(len(at.exception), 0)
        next(r for r in at.radio if r.label == "검토 판단").set_value("실제 이상")
        next(t for t in at.text_area if t.label == "검토 메모").set_value("원천 로그와 대조 완료")
        next(b for b in at.button if b.label == "검토 기록 저장").click().run()
        self.assertEqual(len(at.exception), 0)
        records = at.session_state["review_records"]
        original_id = next(iter(records))
        self.assertEqual(len(records[original_id]), 1)
        self.assertEqual(next(iter(records[original_id].values()))["decision"], "실제 이상")
        next(s for s in at.selectbox if s.label == "계절 주기 설정").set_value("직접 지정").run()
        next(n for n in at.number_input if n.label == "계절 주기 (관측치 수)").set_value(14).run()
        self.assertEqual(len(at.exception), 0)
        records = at.session_state["review_records"]
        self.assertEqual(len(records[original_id]), 1)
        self.assertTrue(any(not value for key, value in records.items() if key != original_id))


if __name__ == "__main__":
    unittest.main()
