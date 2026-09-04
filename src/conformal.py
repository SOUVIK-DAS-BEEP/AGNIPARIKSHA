"""
AGNIPARIKSHA — Stage 7: Conformal Prediction.

Provides distribution-free coverage guarantees on the 168h ML forecasts.
This makes the flag defensible rather than heuristic: we flag when the
statistically guaranteed 95% upper bound exceeds the spec limit.

Key design constraints (SPEC.md Section 4, Stage 7):
  - Use mapie.regression.MapieRegressor with method="plus" and cv=5.
  - The decision uses the UPPER BOUND, not the point prediction.
  - In aerospace, a false negative ends a mission, a false positive
    costs one chip. Be conservative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.regression import MapieRegressor
from typing import Any

from src.constants import COL_I_LEAK, SPEC_LIMITS


def fit_conformal(base_model: Any, X_train: pd.DataFrame, y_train: np.ndarray) -> MapieRegressor:
    """
    Fit MAPIE conformal predictor using cross-validation.
    
    Using cv=5 allows MAPIE to use the entire training set for both
    training the base estimators and computing out-of-fold nonconformity
    scores (cross-conformal prediction). This guarantees the calibration
    data is genuinely held out from each fold's base model.
    """
    # mapie needs an unfitted estimator if we use cv=5, or we can pass
    # the base_model (which might be pre-trained, but mapie will clone it
    # for the CV folds).
    mapie = MapieRegressor(estimator=base_model, method="plus", cv=5, n_jobs=-1)
    mapie.fit(X_train, y_train)
    return mapie


def predict_with_interval(mapie: MapieRegressor, X: pd.DataFrame, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict 168h values with conformal prediction intervals.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (y_pred, y_lower, y_upper)
    """
    y_pred, y_pis = mapie.predict(X, alpha=alpha)
    # y_pis shape is (n_samples, 2, n_alphas). We only have 1 alpha.
    y_lower = y_pis[:, 0, 0]
    y_upper = y_pis[:, 1, 0]
    return y_pred, y_lower, y_upper


def apply_conformal(
    df_wide: pd.DataFrame, 
    X_all: pd.DataFrame, 
    X_train_survivors: pd.DataFrame, 
    y_train_survivors: np.ndarray,
    base_model: Any,
    alpha: float = 0.05
) -> tuple[pd.DataFrame, MapieRegressor]:
    """
    Train conformal predictor and apply to all chips.
    Returns the updated dataframe and the fitted MAPIE model (useful for evaluation).
    """
    df = df_wide.copy()
    spec_max = SPEC_LIMITS[COL_I_LEAK]["max"]
    
    mapie = fit_conformal(base_model, X_train_survivors, y_train_survivors)
    y_pred, y_lower, y_upper = predict_with_interval(mapie, X_all, alpha=alpha)
    
    df["pred_168h"] = y_pred
    df[f"pred_lower_{int((1-alpha)*100)}"] = y_lower
    df[f"pred_upper_{int((1-alpha)*100)}"] = y_upper
    
    # DECISION: Uses the UPPER BOUND, not the point prediction
    df["conformal_flag"] = y_upper > spec_max
    
    return df, mapie
