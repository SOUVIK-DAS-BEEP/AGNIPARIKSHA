"""
AGNIPARIKSHA — Stage 5: ML Drift Forecast.

Predicts the 168h leakage value from early (0h, 24h) measurements plus
physics-based drift rates.

Key design constraints (SPEC.md Section 4, Stage 5):
  - NO DATA LEAKAGE: Features must not include or derive from 96h or 168h
    measurements.
  - TRAIN ON ALL CHIPS: pat_flag uses 168h data and is not available
    at deployment. Filtering to PAT survivors before training is
    selection leakage — the model would be trained on a population
    that cannot be identified at inference time.
  - GroupKFold on lot_id: A random split would leak lot-level info.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any

from src.constants import (
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    COL_TEMP,
    COL_VOLTAGE,
    COL_LOT_ID,
)


def build_features(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Build the exact feature matrix specified in SPEC.md.
    
    Features:
    - i_leak_ua_0h, i_leak_ua_24h, delta_i_leak_0_24
    - v_th_v_0h, v_th_v_24h, delta_v_th_0_24
    - t_delay_ns_0h, t_delay_ns_24h, delta_t_delay_0_24
    - drift_rate_a_i_leak_ua
    - lot_median_i_leak_0h, lot_std_i_leak_0h
    - temp_c, voltage_v
    
    NEVER includes any 96h or 168h data.
    """
    df = df_wide.copy()
    features = pd.DataFrame(index=df.index)
    
    # Base readings at 0h, 24h
    features[f"{COL_I_LEAK}_0h"] = df[f"{COL_I_LEAK}_0h"]
    features[f"{COL_I_LEAK}_24h"] = df[f"{COL_I_LEAK}_24h"]
    features[f"delta_{COL_I_LEAK}_0_24"] = df[f"{COL_I_LEAK}_24h"] - df[f"{COL_I_LEAK}_0h"]
    
    features[f"{COL_V_TH}_0h"] = df[f"{COL_V_TH}_0h"]
    features[f"{COL_V_TH}_24h"] = df[f"{COL_V_TH}_24h"]
    features[f"delta_{COL_V_TH}_0_24"] = df[f"{COL_V_TH}_24h"] - df[f"{COL_V_TH}_0h"]
    
    features[f"{COL_T_DELAY}_0h"] = df[f"{COL_T_DELAY}_0h"]
    features[f"{COL_T_DELAY}_24h"] = df[f"{COL_T_DELAY}_24h"]
    features[f"delta_{COL_T_DELAY}_0_24"] = df[f"{COL_T_DELAY}_24h"] - df[f"{COL_T_DELAY}_0h"]
    
    # Physics anchor
    features[f"drift_rate_a_{COL_I_LEAK}"] = df[f"drift_rate_a_{COL_I_LEAK}"]
    
    # Lot-level aggregates (using 0h ONLY)
    lot_stats = df.groupby(COL_LOT_ID)[f"{COL_I_LEAK}_0h"].agg(["median", "std"]).reset_index()
    lot_stats = lot_stats.rename(columns={"median": "lot_median_i_leak_0h", "std": "lot_std_i_leak_0h"})
    # Handle NaN std if lot has only 1 chip
    lot_stats["lot_std_i_leak_0h"] = lot_stats["lot_std_i_leak_0h"].fillna(0.0)
    
    # Map back to individual chips
    # Use pandas merge/map cautiously to preserve index
    features["lot_median_i_leak_0h"] = df[COL_LOT_ID].map(lot_stats.set_index(COL_LOT_ID)["lot_median_i_leak_0h"])
    features["lot_std_i_leak_0h"] = df[COL_LOT_ID].map(lot_stats.set_index(COL_LOT_ID)["lot_std_i_leak_0h"])
    
    # Stress conditions
    features[COL_TEMP] = df[COL_TEMP]
    features[COL_VOLTAGE] = df[COL_VOLTAGE]
    
    # Ensure exact column order specified (optional but good for consistency)
    col_order = [
        f"{COL_I_LEAK}_0h", f"{COL_I_LEAK}_24h", f"delta_{COL_I_LEAK}_0_24",
        f"{COL_V_TH}_0h", f"{COL_V_TH}_24h", f"delta_{COL_V_TH}_0_24",
        f"{COL_T_DELAY}_0h", f"{COL_T_DELAY}_24h", f"delta_{COL_T_DELAY}_0_24",
        f"drift_rate_a_{COL_I_LEAK}",
        "lot_median_i_leak_0h", "lot_std_i_leak_0h",
        COL_TEMP, COL_VOLTAGE
    ]
    
    return features[col_order]


def train_forecaster(X_train: pd.DataFrame, y_train: np.ndarray | pd.Series, model_type: str = "rf") -> Any:
    """
    Train the forecasting model.
    """
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    elif model_type == "svr":
        model = SVR(kernel="rbf")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    model.fit(X_train, y_train)
    return model


def predict_168h(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Predict 168h leakage using the trained model.
    """
    return model.predict(X)


def evaluate_forecaster_cv(df_wide: pd.DataFrame, model_type: str = "rf") -> dict[str, float]:
    """
    Evaluate the forecaster using GroupKFold on lot_id.
    Trains on ALL chips — pat_flag uses 168h data and is not available
    at deployment.
    
    Returns metrics dict.
    """
    # Train on all chips — pat_flag uses 168h data and is
    # not available at deployment.
    y = df_wide[f"{COL_I_LEAK}_168h"].values
    X = build_features(df_wide)
    groups = df_wide[COL_LOT_ID].values
    
    # Report training target distribution
    print(f"--- Training Target Distribution (all chips) ---")
    print(f"Target: {COL_I_LEAK}_168h")
    print(f"Min:    {np.min(y):.2f} µA")
    print(f"Median: {np.median(y):.2f} µA")
    print(f"Max:    {np.max(y):.2f} µA")
    print(f"---------------------------------------------------------")
    
    # 4. GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    
    y_true_all = []
    y_pred_all = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = train_forecaster(X_train, y_train, model_type)
        preds = predict_168h(model, X_val)
        
        y_true_all.extend(y_val)
        y_pred_all.extend(preds)
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)
    
    return {"mae": mae, "rmse": rmse, "r2": r2}


def apply_forecast(df_wide: pd.DataFrame, model_type: str = "rf") -> pd.DataFrame:
    """
    Train on all chips and predict for all chips.
    This creates the 'pred_168h' column.
    
    Train on all chips — pat_flag uses 168h data and is not available
    at deployment.
    """
    y_train = df_wide[f"{COL_I_LEAK}_168h"].values
    X_train = build_features(df_wide)
    
    # 2. Train model
    model = train_forecaster(X_train, y_train, model_type)
    
    # 3. Predict for ALL chips (we still want forecasts for early-fails to show what would happen)
    X_all = build_features(df_wide)
    preds = predict_168h(model, X_all)
    
    df_res = df_wide.copy()
    df_res["pred_168h"] = preds
    return df_res
