# hotel-demand-forecasting
In this project I was required to forecaste daily hotel demand for 17 hotel properties accross a 28-day period. I used a comprehensive multi-model benchworking framework <www.google.com> | [Notebook 1](www.google.com)

📋 PROJECT OVERVIEW

Datasetsample_hotels.parquet — 17 hotel series, daily frequency

Target Normalised daily room demand (y)

Forecast horizon -> `h = 28 days` (next 4 weeks)

Training period -> `2022-01-01 → 2023-06-02`

Test period -> `2023-06-03 → 2023-06-30`

Total observations -> `10,172 rows across 17 hotels`

Validation strategy -> 5-fold non-overlapping time-series cross-validation (step = 28)

# 🤖 Models Compared

| Category | Model | Package | Description |
|---|---|---|---|
| Baseline | Naive | `statsforecast` | Repeats the last observed value for all future steps |
| Baseline | SeasonalNaive | `statsforecast` | Repeats the last observed seasonal cycle (weekly, season=7) |
| Statistical | AutoETS | `statsforecast` | Automatically selects the best Error/Trend/Seasonality model |
| Statistical | AutoARIMA | `statsforecast` | Automatically selects the best ARIMA order via stepwise search |
| ML | LightGBM | `mlforecast` | Gradient boosted trees with lag, rolling, and calendar features |
| Neural | NBEATS | `neuralforecast` | Neural basis expansion network for interpretable forecasting |
| Neural | NHITS | `neuralforecast` | Neural hierarchical interpolation for long-horizon forecasting |
| Foundation | Chronos T5-small | `chronos-forecasting` | Pretrained language-model-style forecaster, applied zero-shot |


🔧 Methodology

## 🔧 Methodology

### Cross-Validation
Rather than a single train/test split, I used 5-fold time-series cross-validation with non-overlapping windows. Each fold moves forward by 28 days, so the model is tested on 5 different 28-day periods before ever touching the final test set. This gives a much more reliable picture of how each model actually performs.

### LightGBM Features
LightGBM needs hand-crafted features since it has no built-in notion of time. I gave it:
- Recent demand values at lags 1, 7, 14, and 28 days
- Rolling averages and standard deviations to capture short-term trends
- Calendar features like day of week, month, and quarter to capture seasonality
- The target was differenced at lags 1 and 7 before training to remove trend and weekly patterns

### Foundation Model
Chronos is a pretrained forecasting model released by Amazon — think of it like a language model but for time series. Importantly, it requires no training on your data and no API key. I loaded it and ran it directly on each hotel series to generate 20 forecast samples, then took the median as the final prediction.

### Data Cleaning
When I ran the data validation checks, `hotel_77` came up with 16 missing dates in its history. Rather than dropping the series entirely, I filled the gaps using linear interpolation so the series stayed continuous. This was necessary because MLForecast strictly requires gap-free daily timestamps.

### A Note on MAPE
MAPE divides by the actual value, which breaks down completely when demand is zero. Since 320 rows in this dataset have zero demand, reporting MAPE would produce undefined or infinite values for those days. I chose to drop it entirely and rely on MAE and RMSE instead, which are more honest metrics for this data.

