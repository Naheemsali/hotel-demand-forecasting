# hotel-demand-forecasting
In this project I was required to forecaste daily hotel demand for 17 hotel properties accross a 28-day period. I used a comprehensive multi-model benchworking framework <www.google.com> | [Notebook 1](www.google.com)

📋 PROJECT OVERVIEW

Datasetsample_hotels.parquet —> 17 hotel series, daily frequency

Target Normalised daily room demand -> (y)

Forecast horizon -> `h = 28 days` (next 4 weeks)

Training period -> `2022-01-01 → 2023-06-02`

Test period -> `2023-06-03 → 2023-06-30`

Total observations -> `8,806 training rows across 17 hotels`

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


## 🔧 Methodology

### Cross-Validation
I ran a 5-fold time-series cross-validation meaning each model was tested on 5 
different 28-day windows, stepping forward through the data without any overlap between 
folds. By the time a model touches the final held-out test set, it has already been 
evaluated across 140 days worth of out-of-sample predictions. This makes the cross-
validation results much more trustworthy than a single split would be.

### Data Cleaning
When I ran data validation checks `hotel 77` and `hotel 28` had near zero demand accross the entire period. They were unnessary to forecast and only skewed the overall results, therefore, I dropped both datasets. 

### On-The-Books (OTB) Features 
The dataset contains 60 OTB columns representing how many rooms were already booked at 1–60 days before each arrival date. This is a wonderful source of forward looking demand signal. I incorporated OTB features into the models that support them, limiting it to `otb_1` through `otb_28` to prevent data leakage.

### Foundation Model
Chronos is a pretrained forecasting model released by Amazon — think of it like a language model but for time series. Importantly, it requires no training on your data and no API key. I loaded it and ran it directly on each hotel series to generate 20 forecast samples, then took the median as the final prediction.


