"""Evidence-based dataset understanding. No external model or data transmission."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


DOMAIN_TERMS = {
    "유통·수요": ("sales", "revenue", "demand", "inventory", "orders", "매출", "판매", "수요", "재고"),
    "제조·센서": ("sensor", "temperature", "pressure", "vibration", "rpm", "온도", "압력", "진동", "센서"),
    "금융": ("stock", "close", "open", "volume", "return", "price", "주가", "거래량", "수익률"),
    "트래픽·서비스": ("visitors", "traffic", "sessions", "clicks", "latency", "방문자", "접속", "클릭", "응답시간"),
    "관광·이동": ("tourism", "tourist", "passengers", "arrivals", "관광", "여행", "승객", "항공"),
    "에너지": ("energy", "power", "electricity", "load", "kwh", "전력", "발전", "에너지"),
}
DOMAIN_MODES = {"유통·수요": "수요/매출", "제조·센서": "운영/센서", "금융": "금융/리스크"}
DOMAIN_ACTIONS = {
    "유통·수요": ("프로모션·가격 변경·품절·휴점", "가격, 프로모션 여부, 품절 기록, 영업일"),
    "제조·센서": ("운전 모드·정비·센서 교정", "장비 ID, 정비 이력, 운전 상태, 센서 단위"),
    "금융": ("기업 공시·배당·액면분할·거래 중단", "종목 식별자, 수정주가 여부, 공시·시장 일정"),
    "트래픽·서비스": ("캠페인·배포·장애·수집 누락", "배포 이력, 캠페인 일정, 오류율, 수집 로그"),
    "관광·이동": ("공휴일·성수기·운항 변경", "지역, 휴일, 방문 목적, 운항 일정"),
    "에너지": ("기온·운영시간·정전·계량기 변경", "기온, 설비 가동시간, 정전 기록, 측정 단위"),
    "일반·미확인": ("원천 기록·수집 오류·운영 일정", "지표 정의, 단위, 데이터 출처, 이벤트 기록"),
}


@dataclass(frozen=True)
class DatasetProfile:
    domain: str
    evidence: tuple[str, ...]
    certainty: str
    company: str
    description: str
    overridden: bool


def infer_profile(df: pd.DataFrame, filename: str = "", description: str = "",
                  company: str = "", override: str = "자동 추정") -> DatasetProfile:
    text = " ".join([filename, *map(str, df.columns), description]).lower()
    hits = {}
    for domain, terms in DOMAIN_TERMS.items():
        matched = tuple(term for term in terms if (
            term in text if not term.isascii()
            else re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text)
        ))
        if matched:
            hits[domain] = matched
    ranked = sorted(hits, key=lambda domain: len(hits[domain]), reverse=True)
    ambiguous = len(ranked) > 1 and len(hits[ranked[0]]) == len(hits[ranked[1]])
    domain = ranked[0] if ranked and not ambiguous else "일반·미확인"
    evidence = hits.get(domain, ())
    certainty = "복수 분야 단서가 같아 판단 보류" if ambiguous else (
        "여러 용어가 일치함" if len(evidence) >= 2 else "단일 용어에 근거한 약한 추정" if evidence else "판단 근거 부족"
    )
    if override != "자동 추정":
        domain, certainty = override, "사용자 지정"
        evidence = hits.get(domain, ())
    return DatasetProfile(domain, evidence, certainty, company.strip() or "미확인", description.strip(), override != "자동 추정")


def build_findings(series: pd.Series, detected: pd.DataFrame, metrics: pd.DataFrame,
                   meta: dict, context: pd.DataFrame, profile: DatasetProfile) -> list[dict[str, str]]:
    """Return measured facts, explicitly unverified hypotheses, and concrete checks."""
    lookup = dict(zip(metrics["지표"], metrics["값"]))
    findings = []
    seasonal = float(lookup.get("계절성 강도", np.nan))
    trend = float(lookup.get("추세 강도", np.nan))
    findings.append({
        "제목": "반복 패턴과 분석 기준",
        "관측 사실": f"추세 강도 {trend:.2f}, 계절성 강도 {seasonal:.2f}." if np.isfinite(seasonal) else "데이터 길이 또는 주기 때문에 추세·계절성 강도를 계산하지 못했습니다.",
        "검토 가설": "반복 일정이 값 변화에 영향을 줄 수 있습니다." if seasonal >= .45 else "반복 패턴을 뚜렷하게 확인하지 못했습니다.",
        "다음 행동": "STL 분해와 실제 운영 주기를 비교하고 계절 주기를 확인하세요.",
    })
    anomalies = detected[detected.is_anomaly].sort_values("anomaly_score", ascending=False)
    checks, additions = DOMAIN_ACTIONS.get(profile.domain, DOMAIN_ACTIONS["일반·미확인"])
    if anomalies.empty:
        fact = "현재 설정에서 이상 후보가 없습니다."
        hypothesis = "정상임이 입증된 것은 아니며, 작은 변화는 탐지되지 않았을 수 있습니다."
    else:
        top = anomalies.iloc[0]
        fact = f"{len(anomalies)}개 후보 중 최고 점수 시점은 {top.timestamp:%Y-%m-%d %H:%M}, 값 {top.value:.2f}, 점수 {top.anomaly_score:.2f}입니다."
        hypothesis = f"{checks}의 영향을 검토할 수 있습니다. 원인은 아직 확인되지 않았습니다."
    findings.append({"제목": "우선 검토할 이상 후보", "관측 사실": fact, "검토 가설": hypothesis,
                     "다음 행동": f"해당 시점의 {checks} 기록과 원본 값을 대조하세요. 추가하면 좋은 데이터: {additions}."})
    notes = []
    for col in context:
        correlation = series.corr(context[col])
        if np.isfinite(correlation):
            notes.append((col, float(correlation)))
    notes.sort(key=lambda item: abs(item[1]), reverse=True)
    if notes:
        col, correlation = notes[0]
        fact = f"선택한 참고 변수 중 {col}의 동시점 Pearson 상관 절댓값이 가장 큽니다(r={correlation:.2f})."
    else:
        fact = "비교할 참고 변수가 없거나 상관을 계산할 수 없습니다."
    findings.append({"제목": "함께 볼 변수와 데이터 품질", "관측 사실": f"{fact} 값 결측 {meta['missing_values']}개, 중복 시점 {meta['duplicates']}개, 주기 변환 후 빈 구간 {meta.get('inserted_missing', 0)}개입니다.",
                     "검토 가설": "동반 변화는 공통 추세나 계절성에서 생길 수 있어 인과관계로 해석할 수 없습니다.",
                     "다음 행동": "원본과 보간 영향을 확인하고, 참고 변수의 차분·시차 관계를 추가 검토하세요."})
    return findings


def question_catalog(profile: DatasetProfile) -> dict[str, tuple[str, str]]:
    checks, additions = DOMAIN_ACTIONS.get(profile.domain, DOMAIN_ACTIONS["일반·미확인"])
    return {
        "어느 시점을 먼저 검토해야 할까?": ("점수가 높은 후보를 먼저 확인하고 주변 시점과 비교합니다.", f"{checks} 기록을 대조하세요."),
        "반복 패턴을 벗어난 변화일까?": ("추세·계절·잔차를 나누어 반복 패턴으로 설명되는지 확인합니다.", "분해 주기는 휴일·운영 일정과 비교해 확정하세요."),
        "분석을 개선하려면 무엇이 더 필요할까?": (f"추가 권장 데이터: {additions}.", "정답 이상치 라벨을 확보하면 실제 데이터의 Precision·Recall·F1을 평가할 수 있습니다."),
    }


def profile_report(profile: DatasetProfile, findings: list[dict[str, str]]) -> str:
    lines = ["# 데이터 이해 및 검토 계획", f"- 분야: {profile.domain} ({profile.certainty})",
             f"- 분야 단서: {', '.join(profile.evidence) or '없음'}", f"- 기업: {profile.company} (사용자 입력 기준)",
             f"- 데이터 설명: {profile.description or '미입력'}", "", "분야 추정은 용어 기반이며 기업 및 원인을 자동 확인하지 않습니다."]
    for finding in findings:
        lines.extend(["", f"## {finding['제목']}", *[f"- {key}: {finding[key]}" for key in ("관측 사실", "검토 가설", "다음 행동")]])
    return "\n".join(lines)
