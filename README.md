# Hotel-Demand-Forecasting
In this project I was required to forecast daily hotel demand for 17 hotel properties across a 28-day period. I used a comprehensive multi-model benchworking framework <[Open in Google Colab](https://colab.research.google.com/drive/https://colab.research.google.com/drive/1dAi5kjSD4AqLHn08wTjalRR2mnPwgkxC?usp=sharing)> | [Notebook 1](www.google.com)

PROJECT OVERVIEW

Dataset `sample_hotels.parquet` —> 17 hotel series, daily frequency

Target `Normalised daily room demand` -> (y)

Forecast horizon -> `h = 28 days` (next 4 weeks)

Training period -> `2022-01-01 → 2023-06-02`

Test period -> `2023-06-03 → 2023-06-30`

Total observations -> `8,806 training rows across 17 hotels`

Validation strategy -> 5-fold non-overlapping time-series cross-validation (step = 28)

#  Models Compared

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


#  Project Workflow

### Data Loading & Preparation 
I began by loading the hotel demand dataset and standardizing the column names to match forecasting library requirements `unique_id`, `ds`, and `y`. The data was sorted chronologically for each hotel to ensure proper time-series structure.

During initial validation, I identified two hotels `hotel_28` and `hotel_77` with near-zero demand across the entire time period. Because these series provided no meaningful signal and distorted evaluation metrics, I removed them from the dataset. The final dataset consists of 17 hotel time series, each with consistent daily observations.

### Data Quality Checks
Before modeling, I performed several validation checks:

- Verified no missing values in key columns `unique_id`, `ds`, `y`
- Confirmed consistent time series lengths across all hotels
- Identified and got rid of duplicate or missing dates by reindexing each series
- Interpolated minor gaps in demand values to maintain continuity

These steps make sure that all models were trained on clean, reliable time-series data. 

### Train/Test Splits
Next, I split the dataset into 2 sets:

- Training set: All observations except the final 28 days
- Test set: The final 28 days for each hotel

The test set was completely held out and only used for final model evaluation, ensuring an unbiased assessment of performance.

### Cross-Validation Strategy 
To evaluate model performance robustly, I implemented a 5-fold rolling time-series cross-validation:

- Each fold forecasts a 28-day horizon (H = 28)
- Folds move forward in time with no overlap
- Total of 140 days of out-of-sample evaluation per model

This approach provides a much more reliable estimate of performance than a single train/test split, as each model is tested across multiple time periods

This allows it to capture complex temporal patterns without manual feature engineering.

### Baseline & Statistical Models 
I first established benchmark models using classical time-series approaches:

- Naive
- Seasonal Naive
- AutoETS
- AutoARIMA

These models were implemented using the StatsForecast library and evaluated across all cross-validation folds.

### Machine Learning Model (LightGBM)
Next, I built a machine learning model using LightGBM 

To enhance predictive power, I incorporated:

- Lag features (1, 7, 14, 28 days)
- Rolling statistics (means and standard deviations)
- Calendar features (day of week, month, quarter)

Additionally, I included On-The-Books (OTB) features, which represent bookings made 1–60 days in advance. To prevent data leakage, I restricted these features to `otb_1` through `otb_28`, ensuring only information available before the forecast horizon was used.


### Neural Forecast Modeling 
I then implemented deep learning models using the NeuralForecast framework:

- NBEATS 
- NHITS 

NHITS was able to leverage OTB features, while NBEATS relied solely on historical demand. Both models were trained and evaluated using the same cross-validation framework for consistency.

### Foundation Model (Chronos) 
Finally, I incorporated Chronos, a pretrained time-series foundation model developed by Amazon.

Key characteristics:

- No dataset-specific training required
- Generates multiple forecast samples
- Final predictions are computed using the median forecast

Chronos was evaluated using a custom 5-fold cross-validation procedure to ensure consistency with other models.

#  Evaluation & Outputs
To compare model performance and ensure reproducibility, multiple outputs were generated throughout the forecasting pipeline

## 1. `Cross-Validation Predictions`

File: cross_validation_predictions.csv

This file contains all predictions from every model across all cross-validation folds.

Each row represents:

- A specific hotel (unique_id)
- A forecast date (ds)
- The actual demand (y)
- Predictions from each model

Purpose: 

Allows full inspection of model behavior over time

Serves as the foundation for computing evaluation metrics

## 2. `Full Metrics (Per Series & Model)`

File: full_metrics.csv

This dataset reports evaluation metrics for each hotel and each model.

Metrics included:

- ME (bias)
- MAE
- RMSE
- MAPE (when applicable)

Purpose:

Shows how each model performs on individual hotel series

Helps identify whether certain models perform better for specific types of hotels

## 3. `Model Summary (Overall Performance)`

File: model_summary.csv

This file aggregates performance across all hotels by averaging metrics.

Purpose:

Provides a high-level comparison of models

Used to determine the best overall model

Interpretation:

Lower RMSE / MAE → better model
ME close to 0 → less bias

## 4. `Model Win Counts`

File: model_wins.csv

This table counts how often each model achieved the lowest error for each metric across all hotels.

Purpose:

Measures consistency across series

Highlights whether a model wins frequently or only occasionally

Interpretation:

More wins = more consistent performance

A model may have best average performance but fewer wins

## 5. `Final Test Forecasts`

File: final_28_day_hotel_forecasts.csv

This file contains the final 28-day demand forecasts for each hotel.

Each row includes:

- Hotel ID
- Forecast date
- Predicted demand

Purpose:

Represents the actual deliverable forecast

Simulates real-world future demand predictions

## 6. `Forecast vs. Actual Plots`

Location: plots/ folder

These visualizations compare:

Actual demand
Model predictions

Purpose:

Provides intuitive understanding of model performance

Highlights:
- Trend tracking
- Seasonality capture
- Forecast error patterns

## Key Findings

The AutoNBEATS and AutoNHITS models were the strongest performers during cross-validation, both achieving the lowest average RMSE of approximately 0.148. This suggests that the automated neural forecasting models were effective at capturing the nonlinear demand patterns and seasonality present across the hotel series.

On the final held-out test set, AutoARIMA achieved the lowest RMSE of approximately 0.112, slightly outperforming AutoNHITS and AutoNBEATS. However, the difference between these three models was small, indicating that both the statistical and neural approaches generalized well to unseen hotel demand data.

The model win counts show that performance varied by hotel and metric rather than one model dominating every category. AutoNBEATS had the strongest cross-validation win profile overall, while Chronos and AutoARIMA were also competitive across several metrics. This suggests that different hotel series may benefit from different model structures.

Chronos performed competitively without dataset-specific training, which highlights the value of pretrained foundation models for forecasting tasks. Although it was not the top model overall, its performance was close to the leading models and required less tuning.

LightGBM underperformed relative to the time-series-specific models, even after incorporating OTB features. This suggests that feature-based machine learning approaches may struggle to capture sequential demand patterns compared with models designed specifically for forecasting.

Overall, the results show that AutoNBEATS and AutoNHITS were strongest in cross-validation, while AutoARIMA was the best final test performer. Because the top models were close, the final takeaway is that both automated neural forecasting models and traditional statistical models are highly effective for this hotel demand forecasting problem.

