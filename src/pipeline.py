"""
AGNIPARIKSHA — Master Pipeline.

Orchestrates Stages 1 through 8 as a single entry point.
"""

from __future__ import annotations

import pandas as pd
from typing import Any

from src.constants import COL_I_LEAK
from src.ingest import validate_schema, to_wide
from src.pat import pat_screen
from src.multivariate import mahalanobis_screen
from src.arrhenius import apply_arrhenius_fit
from src.forecast import build_features, train_forecaster, predict_168h
from src.lifetime import apply_lifetime_projection
from src.conformal import apply_conformal
from src.decide import make_decision
from src.evaluate import evaluate, EvalReport


def run_pipeline(
    df_long: pd.DataFrame, 
    alpha: float = 0.05, 
    model_type: str = "rf"
) -> tuple[pd.DataFrame, Any, EvalReport | None]:
    """
    Run the full AGNIPARIKSHA pipeline end-to-end.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Raw burn-in data in the canonical long format.
    alpha : float
        Conformal prediction significance level (default 0.05 for 95% coverage).
    model_type : str
        Model to use for forecasting ("rf" or "svr").
        
    Returns
    -------
    tuple[pd.DataFrame, Any, EvalReport | None]
        - Wide-format dataframe containing all features, predictions, and flags.
        - The fitted MapieRegressor model (for explanations).
        - Evaluation report if true_class was present, else None.
    """
    # 1. Ingest
    val_report = validate_schema(df_long)
    if not val_report.is_valid:
        raise ValueError(f"Schema validation failed:\n{val_report}")
    df = to_wide(df_long)
    
    # 2. PAT Screen
    df = pat_screen(df)
    
    # 3. Multivariate Screen
    df = mahalanobis_screen(df)
    
    # 4. Arrhenius Fit
    df = apply_arrhenius_fit(df)
    
    # 5. ML Forecast (Train on PAT survivors only)
    survivors = df[~df["pat_flag"]].copy()
    if len(survivors) == 0:
        raise RuntimeError("No chips survived PAT. Cannot train forecaster.")
        
    X_train = build_features(survivors)
    y_train = survivors[f"{COL_I_LEAK}_168h"].values
    
    base_model = train_forecaster(X_train, y_train, model_type)
    
    X_all = build_features(df)
    
    # 6. Lifetime Projection
    df = apply_lifetime_projection(df)
    
    # 7. Conformal Prediction
    # apply_conformal computes pred_168h (point prediction) and the bounds,
    # and sets the conformal_flag.
    df, mapie_model = apply_conformal(
        df_wide=df, 
        X_all=X_all, 
        X_train_survivors=X_train, 
        y_train_survivors=y_train, 
        base_model=base_model, 
        alpha=alpha
    )
    
    # 8. Decision
    df = make_decision(df)
    
    # Evaluation (if ground truth available)
    eval_report = None
    if "true_class" in df.columns:
        eval_report = evaluate(df, alpha=alpha)
        
    return df, mapie_model, eval_report
