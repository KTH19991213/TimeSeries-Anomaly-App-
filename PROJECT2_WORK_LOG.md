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
