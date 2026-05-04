# hotel-demand-forecasting
In this project I was required to forecaste daily hotel demand for 19 hotel properties accross a 28-day period. I used a comprehensive multi-model benchworking framework 

📋 PROJECT OVERVIEW

Datasetsample_hotels.parquet — 19 hotel series, daily frequency

Target Normalised daily room demand (y ∈ [0, 1])

Forecast horizon -> h = 28 days (next 4 weeks)

Training period -> 2022-01-01 → 2023-06-02 

Test period -> 2023-06-03 → 2023-06-30

Total observations -> 10,172 rows across 19 hotels

Validation strategy -> 5-fold non-overlapping time-series cross-validation (step = 28)

🤖 Models Compared

## 🤖 Models Compared

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


