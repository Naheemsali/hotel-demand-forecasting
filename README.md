# Hotel-Demand-Dorecasting
In this project I was required to forecaste daily hotel demand for 17 hotel properties accross a 28-day period. I used a comprehensive multi-model benchworking framework <www.google.com> | [Notebook 1](www.google.com)

PROJECT OVERVIEW

Datasetsample_hotels.parquet —> 17 hotel series, daily frequency

Target Normalised daily room demand -> (y)

Forecast horizon -> `h = 28 days` (next 4 weeks)

Training period -> `2022-01-01 → 2023-06-02`

Test period -> `2023-06-03 → 2023-06-30`

Total observations -> `8,806 training rows across 17 hotels`

Validation strategy -> 5-fold non-overlapping time-series cross-validation (step = 28)

##  Models Compared

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


##  Methodology

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

### Foundation Model (Chronos)
Chronos is a pretrained time-series foundation model developed by Amazon. It operates similarly to a language model but is designed for forecasting tasks.

Unlike traditional models, Chronos:

Requires no training on the dataset
Generates multiple forecast samples
Produces predictions using the median of those samples

This allows it to capture complex temporal patterns without manual feature engineering.

##  Evaluation & Outputs
To compare model performance and ensure reproducibility, multiple outputs were generated throughout the forecasting pipeline

1. Cross-Validation Predictions

File: cross_validation_predictions.csv

This file contains all predictions from every model across all cross-validation folds.

Each row represents:

A specific hotel (unique_id)
A forecast date (ds)
The actual demand (y)
Predictions from each model

 Purpose: 

Allows full inspection of model behavior over time

Serves as the foundation for computing evaluation metrics

2. Full Metrics (Per Series & Model)

File: full_metrics.csv

This dataset reports evaluation metrics for each hotel and each model.

Metrics included:

ME (bias)
MAE
RMSE
MAPE (when applicable)

Purpose:

Shows how each model performs on individual hotel series
Helps identify whether certain models perform better for specific types of hotels

3. Model Summary (Overall Performance)

File: model_summary.csv

This file aggregates performance across all hotels by averaging metrics.

Purpose:

Provides a high-level comparison of models
Used to determine the best overall model

Interpretation:

Lower RMSE / MAE → better model
ME close to 0 → less bias
4. Model Win Counts

File: model_wins.csv

This table counts how often each model achieved the lowest error for each metric across all hotels.

Purpose:

Measures consistency across series
Highlights whether a model wins frequently or only occasionally

Interpretation:

More wins = more consistent performance
A model may have best average performance but fewer wins
5. Final Test Forecasts

File: final_28_day_hotel_forecasts.csv

This file contains the final 28-day demand forecasts for each hotel.

Each row includes:

Hotel ID
Forecast date
Predicted demand

Purpose:

Represents the actual deliverable forecast
Simulates real-world future demand predictions
6. Forecast vs. Actual Plots

Location: plots/ folder

These visualizations compare:

Actual demand
Model predictions

Purpose:

Provides intuitive understanding of model performance
Highlights:
Trend tracking
Seasonality capture
Forecast error patterns

## Key Findings 



