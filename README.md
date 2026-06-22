# 시계열 이상탐지 분석 대시보드

시계열분석 프로젝트 2 제출용 Streamlit 웹앱입니다.



## 목표

다변량 CSV 파일을 업로드하면 날짜/시간 컬럼과 수치형 컬럼을 자동으로 후보 탐지하고, 사용자가 선택한 대상 시계열에 대해 이상치를 자동 탐지합니다. 단순히 이상치를 표시하는 데서 끝나지 않고, 탐지 근거, 이상치 유형, 시각화 해석, 데이터 품질 점검, 권장 검토 조치까지 함께 보여주는 분석 대시보드입니다.

웹앱 소개 동영상은 별도로 직접 녹화할 예정이므로 이 저장소에는 포함하지 않습니다.

## 주요 기능

- CSV 업로드 및 날짜/수치 컬럼 자동 인식
- 다변량 CSV의 추가 수치 컬럼을 참고 변수로 선택 가능
- 파일, 컬럼, 주기, 탐지 설정 변경 시 자동 재분석
- 업로드 데이터별 자동 분석 브리핑 제공
- 민트/코랄 톤의 카드형 대시보드 UI 제공
- 점수 추이, 점수 분포, 탐지 기준별 점수에 대한 시각화 해석 가이드 제공
- 결측치, 중복 시점, 시간 주기, 정상성, 이상치 비율에 대한 데이터 품질 점검
- 강의 기반 시계열 진단 제공
  - 원시 시계열과 1차 차분의 ADF 정상성 검정 비교
  - ACF/PACF 자기상관 진단
  - STL 추세/계절/잔차 분해
- 앙상블 이상탐지 방식
  - rolling z-score
  - rolling IQR distance
  - STL residual score
  - multifeature Isolation Forest score
- 시계열 그래프, 이상 점수, 점수 분포, 탐지 기준별 점수, 이상치 유형 분포 시각화
- 이상치별 유형, 탐지 근거, 참고 변수 해석, 권장 검토 조치 제공
- 주요 평가 지표 제공
  - 이상치 개수와 비율
  - 평균, 표준편차, 변동계수
  - 보간 전 결측치 수
  - 1시차 자기상관
  - ADF p-value
  - 추세 강도와 계절성 강도
- 이상탐지 결과 CSV 다운로드
- 분석 리포트 Markdown 다운로드
- 업로드 파일이 없어도 실행 가능한 내장 샘플 데이터 제공

## 로컬 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

`python app.py`로 실행해도 자동으로 Streamlit 실행으로 전환되도록 처리했습니다. 그래도 시연과 배포 전 확인에는 `streamlit run app.py` 명령을 권장합니다.


## 권장 CSV 형식

앱은 다양한 CSV 형식을 허용하지만, 아래와 같은 구조에서 가장 안정적으로 작동합니다.

```csv
date,value
2026-01-01,72.1
2026-01-02,73.4
```

다변량 CSV의 경우 업로드 후 사이드바에서 분석 대상 수치 컬럼을 선택하고, 필요하다면 추가 수치 컬럼을 참고 변수로 선택하면 됩니다.

## 시연용 다변량 CSV

녹화 시연에는 `date_count_multivariate.csv` 파일을 사용할 수 있습니다. 이 파일은 원본 `date_count.csv`에 아래 참고 변수를 추가한 다변량 시계열 CSV입니다.

- `day_of_week`
- `is_weekend`
- `lag_1`
- `diff_1`
- `rolling_mean_7`
- `rolling_std_7`

시연에서는 분석 대상 값 컬럼을 `count`로 선택하고, 참고 변수에는 위 컬럼들을 선택하면 다변량 CSV 분석 흐름을 보여줄 수 있습니다.

추가로 Darts 공식 예제 데이터 기반의 `australian_tourism_monthly.csv`도 포함했습니다.

- 출처: https://unit8co.github.io/darts/generated_api/darts.datasets.datasets.html
- 데이터셋: `AustralianTourismDataset`
- 구조: `Month`와 96개 수치형 관광 지표
- 기간: 월별 36개 관측치

이 파일은 지역, 방문 목적, 도시/비도시 구분이 함께 들어 있는 명확한 다변량 시계열 CSV입니다. 시연에서는 날짜 컬럼을 `Month`, 분석 대상 값 컬럼을 `Total`, 참고 변수는 `Hol`, `VFR`, `Bus`, `Oth`, `NSW`, `VIC`, `QLD`, `SA`, `WA`, `TAS`, `NT`처럼 여러 개 선택하면 다변량 CSV 분석 흐름을 분명하게 보여줄 수 있습니다.
