# Project 2 Work Log

## Assignment Requirements

The attached assignment image asks for a web app that:

- uploads an arbitrary time series CSV file, including multivariate CSV data
- automatically analyzes the data and performs time series anomaly detection
- can collect new anomaly detection results when the file changes
- provides a visualization dashboard with several evaluation indicators
- is deployed online with a link
- includes a 3-minute web app introduction video

The video item is excluded because the user said they will record it directly.

## Context Referenced From This Folder

The folder already contained earlier conversation records and project notes:

- `chatgpt_share_69bfb280_extracted.md`
- `졸업프로젝트_대화흐름_분석기록.md`
- existing Streamlit forecasting app files such as `time_series_app.py` and `src/*`

Those records mainly describe a previous recommendation-system project direction. For Project 2, the implementation was redirected to the current time series analysis assignment: CSV upload, automated anomaly detection, dashboard indicators, and deployable Streamlit app packaging.

## Implementation Decisions

- Streamlit was selected because the folder already used Streamlit and it is easy to deploy online.
- The app supports arbitrary CSV files by detecting datetime-like and numeric-like columns.
- The anomaly score is an ensemble score instead of a single rule:
  - rolling z-score captures local sudden changes
  - rolling IQR distance captures robust outliers
  - STL residual score captures trend/seasonality-adjusted anomalies
  - Isolation Forest captures multifeature unusual points
- The app includes an interpretation layer:
  - anomaly type classification
  - method agreement count
  - context-column root-cause hints
  - reason text for each anomaly
  - recommended analyst action by business mode
  - an automatic briefing for each uploaded dataset
- A built-in sample dataset is included inside the app, so the app can be demonstrated even before uploading a CSV.
- Export buttons were added for anomaly results and a markdown report.

## Files Created

- `app.py`: main Streamlit web application
- `requirements.txt`: deploy/runtime dependencies
- `README.md`: run, feature, and deployment instructions
- `PROJECT2_WORK_LOG.md`: this record

## Notes For Submission

To complete the online submission:

1. Push this folder to a GitHub repository under `KTH19991213`.
2. Deploy it on Streamlit Community Cloud.
3. Submit the deployed URL.
4. Record the web app introduction video separately.

## 2026-10-16 Portfolio Revision

- Added evidence-based industry inference, explicit abstention, user correction and user-supplied organization metadata.
- Added dataset understanding, three findings with fact/hypothesis/action separation, question-driven charts and downloadable review plans.
- Added monthly/day/hour aggregation choices, aligned context resampling, time-based interpolation, gap counts, encoding fallback and expansion guards.
- Fixed anomaly/severity consistency and normalized weights of active, available methods.
- Fixed near-zero STL residual MAD causing astronomical scores on the bundled tourism CSV; fall back to standard deviation at floating-point precision limits.
- Added nine seeded synthetic cases and a standalone z-score baseline; published raw point-level results including weaknesses on sustained level changes and normal-pattern false alarms.
- Updated portfolio README with design decisions, reproducibility and limits. This revision is local; the GitHub deployment has not been changed.

## 2026-10-17 Upload Button Contrast Fix

- Replaced the uploader's dark text override with explicit white text on a dark background, including hover and keyboard-focus styles.
- Verified computed browser styles: text rgb(255,255,255), background rgb(52,51,66); saved docs/upload_button_fixed.png.
- Next portfolio priorities: anomaly drill-down with threshold evidence, review labels with dataset/settings identity, level-shift detection, configurable seasonal periods, and reproducible analysis manifests.

## 2026-10-17 Review Workflow and Level Shift Revision

- Added anomaly drill-down with neighborhood plot, local median baseline, selected candidate, ensemble threshold, method votes and level-shift evidence.
- Added per-analysis review forms and session records, portable JSON export/import, file/settings identity checks, and review columns in result CSV/Markdown.
- Added retrospective sustained-level segmentation with robust slope/seasonal adjustment and optional enable switch. Added manual seasonal period propagated to detection and diagnostics.
- Added SHA-256 dataset identity, analysis identity and downloadable preprocessing/configuration/environment manifest.
- Compared three methods across nine synthetic cases (27 raw rows). Sustained-shift mean F1: previous 0.260, added method 0.788; mean false positives: 5.667 to 6.667. No claim of real-world accuracy.
- Twenty automated tests passed, including extra seeds 127 and 2026, review/manifest roundtrip and mismatched-analysis rejection.
- Browser verified review save, JSON download and reload/import workflow. Deployment remains unchanged; source and ZIP are local.

## 2026-10-17 GitHub and Streamlit Release

- Published the portfolio revision to GitHub main and verified the public Streamlit URL.
- Initial exact local dependency pins failed to build older PyArrow/NumPy on the cloud Python 3.14.8 runtime. Restored compatible ranges and rebooted from the authenticated management panel; dependency installation and app startup succeeded.
- Verified deployed dataset understanding, anomaly drill-down, review save, manifest download and 27-row benchmark execution. Deployed benchmark means match the documented local results.
- Added current Streamlit React Aria sidebar styling so select arrows and multiselect controls retain contrast on the dark sidebar.
