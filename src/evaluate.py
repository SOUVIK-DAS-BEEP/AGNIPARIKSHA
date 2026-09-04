"""
AGNIPARIKSHA — Section 6: Evaluation.

Computes the core metrics demonstrating the pipeline's value, notably
latent-defect recall for both the baseline (PAT) and the full pipeline.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, precision_recall_curve
)
from dataclasses import dataclass
from typing import Any

from src.constants import CLASS_LATENT, CLASS_EARLY_FAIL, COL_I_LEAK


@dataclass
class EvalReport:
    """Holds the results of the evaluation."""
    pat_recall_latent: float
    pat_precision: float
    pat_f1: float
    
    full_recall_latent: float
    full_precision: float
    full_f1: float
    full_roc_auc: float
    
    conformal_coverage: float
    
    pat_conf_matrix: np.ndarray
    full_conf_matrix: np.ndarray
    
    pr_curve_precisions: np.ndarray
    pr_curve_recalls: np.ndarray
    pr_curve_thresholds: np.ndarray


def evaluate(df_results: pd.DataFrame, alpha: float = 0.05) -> EvalReport:
    """
    Evaluate pipeline performance vs baseline (PAT).
    
    Optimises for RECALL on latent-defective chips (the headline metric).
    """
    df = df_results.copy()
    
    if "true_class" not in df.columns:
        raise ValueError("true_class column required for evaluation.")
        
    # We consider BOTH latent and early_fail as "positive" (defective) for general metrics
    y_true = (df["true_class"] == CLASS_LATENT) | (df["true_class"] == CLASS_EARLY_FAIL)
    
    # 1. Baseline (PAT) Metrics
    y_pred_pat = df["pat_flag"]
    
    # Latent-specific recall (the most important metric)
    latent_mask = (df["true_class"] == CLASS_LATENT)
    if latent_mask.sum() > 0:
        pat_recall_latent = df.loc[latent_mask, "pat_flag"].mean()
        full_recall_latent = df.loc[latent_mask, "verdict"].apply(lambda v: v == "FLAG").mean()
    else:
        pat_recall_latent = np.nan
        full_recall_latent = np.nan
        
    pat_prec = precision_score(y_true, y_pred_pat, zero_division=0)
    pat_f1 = f1_score(y_true, y_pred_pat, zero_division=0)
    pat_cm = confusion_matrix(y_true, y_pred_pat)
    
    # 2. Full Pipeline Metrics
    y_pred_full = df["verdict"] == "FLAG"
    
    full_prec = precision_score(y_true, y_pred_full, zero_division=0)
    full_f1 = f1_score(y_true, y_pred_full, zero_division=0)
    full_cm = confusion_matrix(y_true, y_pred_full)
    
    # For ROC and PR curve, we use the risk_score (0-4) as a pseudo-probability
    # Note: risk_score is highly discrete, but it serves the purpose.
    y_score = df["risk_score"] / 4.0
    full_roc_auc = roc_auc_score(y_true, y_score)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    
    # 3. Conformal Coverage
    # True value is 168h leakage
    # Covered if true value is between lower and upper bounds
    y_168 = df[f"{COL_I_LEAK}_168h"]
    lower_col = f"pred_lower_{int((1-alpha)*100)}"
    upper_col = f"pred_upper_{int((1-alpha)*100)}"
    
    if lower_col in df.columns and upper_col in df.columns:
        covered = (y_168 >= df[lower_col]) & (y_168 <= df[upper_col])
        coverage = covered.mean()
    else:
        coverage = np.nan
        
    return EvalReport(
        pat_recall_latent=pat_recall_latent,
        pat_precision=pat_prec,
        pat_f1=pat_f1,
        full_recall_latent=full_recall_latent,
        full_precision=full_prec,
        full_f1=full_f1,
        full_roc_auc=full_roc_auc,
        conformal_coverage=coverage,
        pat_conf_matrix=pat_cm,
        full_conf_matrix=full_cm,
        pr_curve_precisions=precisions,
        pr_curve_recalls=recalls,
        pr_curve_thresholds=thresholds,
    )
