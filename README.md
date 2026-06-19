# Time Series Anomaly Intelligence

Project 2 submission app for time series analysis.

## Goal

Upload any multivariate CSV, choose a time column and one numeric value column, and automatically detect time series anomalies. The dashboard helps judge whether detected anomalies are reasonable by showing the original series, anomaly points, score distribution, component scores, evaluation indicators, anomaly type, explanation, and recommended review action.

The web app introduction video is intentionally excluded because it will be recorded separately.

## Differentiation Strategy

Project 1 lost points because the differentiation function/UI score was 0. This version directly addresses that feedback.

Most basic submissions can stop at "upload CSV -> detect anomaly -> show chart." This app adds an interpretation layer:

- anomaly type classification: spike/drop, level deviation, volatility burst, seasonality residual, mixed signal
- method agreement count: shows how many detection methods supported the anomaly
- context-column root-cause hints: optional extra numeric columns are used as supporting signals
- action-oriented review board: each anomaly receives a reason and a recommended analyst action
- differentiation tab: explicitly states how the app differs from a baseline anomaly dashboard
- downloadable explanation report, not only a raw result CSV

## Main Features

- CSV upload with automatic datetime and numeric column candidate detection
- Optional context columns for multivariate anomaly interpretation
- Re-analysis whenever a different file, column, frequency, or detection setting is selected
- Ensemble anomaly detection using:
  - rolling z-score
  - rolling IQR distance
  - STL residual score
  - multifeature Isolation Forest score
- Dashboard visualizations for time series, anomaly score, score distribution, method component scores, and anomaly type counts
- Anomaly review table with reason, context note, and recommended action
- Evaluation indicators:
  - anomaly count and rate
  - mean, standard deviation, coefficient of variation
  - missing values before interpolation
  - lag-1 autocorrelation
  - ADF p-value
  - trend and seasonality strength
- CSV export and markdown report export
- Built-in sample data, so the app runs even before a user uploads a file

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If you run the file with `python app.py`, the app now redirects itself to Streamlit automatically. Still, `streamlit run app.py` is the recommended command.

## Online Deployment

Recommended deployment path:

1. Create a GitHub repository under `https://github.com/KTH19991213`.
2. Upload this `project2_anomaly_app` folder.
3. Deploy with Streamlit Community Cloud.
4. Set the app entry point to `app.py`.

## Expected CSV Format

The app accepts flexible CSV files, but it works best with:

```csv
date,value
2026-01-01,72.1
2026-01-02,73.4
```

For multivariate CSV files, upload the file, select the desired numeric target column, and optionally select additional numeric context columns in the sidebar.
