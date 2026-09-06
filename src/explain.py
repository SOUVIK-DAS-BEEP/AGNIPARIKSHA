"""
AGNIPARIKSHA — Stage 8b: Explainability.

Generates SHAP explanations for the ML forecaster to provide local,
per-chip transparency.

Key design constraints (SPEC.md Section 4, Stage 8):
  - Use shap.TreeExplainer on the RandomForest.
  - CACHE the explainer. Do not rebuild it per chip (TreeExplainer
    setup is expensive).
"""

from __future__ import annotations

import pandas as pd
import shap
from typing import Any


# Global cache for the explainer to avoid rebuilding it per chip
_EXPLAINER_CACHE = {}


def get_explainer(model: Any) -> shap.TreeExplainer:
    """
    Get or create a cached TreeExplainer for the given model.
    """
    model_id = id(model)
    if model_id not in _EXPLAINER_CACHE:
        # Assuming the model is a RandomForest or similar tree ensemble.
        # If MAPIE wraps it, we must extract the base estimator.
        base = model
        if hasattr(model, "estimator_"):
            # MAPIE >= 0.8 uses an EnsembleRegressor in estimator_
            if hasattr(model.estimator_, "single_estimator_"):
                base = model.estimator_.single_estimator_
            else:
                base = model.estimator_
        elif hasattr(model, "single_estimator_"):
            # Older MAPIE versions
            base = model.single_estimator_
            
        _EXPLAINER_CACHE[model_id] = shap.TreeExplainer(base)
        
    return _EXPLAINER_CACHE[model_id]


def explain_chip(model: Any, X_row: pd.DataFrame, feature_names: list[str]) -> dict:
    """
    Generate SHAP explanation for a single chip.
    
    Parameters
    ----------
    model : Any
        The trained RandomForest (or MapieRegressor wrapping it).
    X_row : pd.DataFrame
        A single-row DataFrame containing the features for one chip.
    feature_names : list[str]
        List of feature names.
        
    Returns
    -------
    dict
        Dictionary containing the base value, shap values, and feature values
        ready for plotting in Streamlit.
    """
    explainer = get_explainer(model)
    
    # Calculate SHAP values for the single row
    shap_values = explainer(X_row)
    
    return {
        "base_value": float(shap_values.base_values[0]),
        "shap_values": shap_values.values[0].tolist(),
        "data": X_row.iloc[0].tolist(),
        "feature_names": feature_names,
    }
