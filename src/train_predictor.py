"""
Train an XGBoost model to predict daily max temperature from morning METAR observations.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json


FEATURE_COLS = [
    'morning_temp',
    'morning_temp_max',
    'morning_dewpoint_depression',
    'morning_wind_speed',
    'morning_gust',
    'morning_wind_u',
    'morning_wind_v',
    'morning_cloud_score',
    'morning_cloud_base',
    'morning_precip',
    'morning_pressure',
    'day_of_year',
    'prev_day_max',
    'pressure_tendency',
    'temp_trend',
]

TARGET = 'max_temp'


def train_model(dataset_path=None):
    """Train XGBoost model with time-series cross-validation."""
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")

    df = pd.read_csv(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Dataset: {len(df)} samples, {df['date'].min().date()} to {df['date'].max().date()}")

    # Handle missing/infinite values
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    print("\nTime-Series Cross-Validation:")
    print("-" * 50)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"  Fold {fold+1}: MAE={mae:.2f}°C  RMSE={rmse:.2f}°C  R²={r2:.3f}")

    print("-" * 50)
    print(f"  Mean:  MAE={np.mean(mae_scores):.2f}°C  RMSE={np.mean(rmse_scores):.2f}°C  R²={np.mean(r2_scores):.3f}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    final_model.fit(X, y, verbose=False)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, final_model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("\nFeature Importance:")
    for feat, imp in importance.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:35s} {imp:.3f} {bar}")

    # Save model and metadata
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "max_temp_predictor.json")
    final_model.save_model(model_path)

    # Save feature medians for filling missing values at inference
    medians = X.median().to_dict()
    medians_path = os.path.join(model_dir, "feature_medians.json")
    with open(medians_path, "w") as f:
        json.dump(medians, f, indent=2)

    metadata = {
        "station": "LTAC",
        "features": FEATURE_COLS,
        "target": TARGET,
        "cv_mae": round(float(np.mean(mae_scores)), 2),
        "cv_rmse": round(float(np.mean(rmse_scores)), 2),
        "cv_r2": round(float(np.mean(r2_scores)), 3),
        "n_samples": len(df),
        "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
        "importance": {k: round(float(v), 4) for k, v in importance.items()},
    }
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {model_path}")
    print(f"Metadata saved to {meta_path}")

    return final_model, metadata


if __name__ == "__main__":
    train_model()
