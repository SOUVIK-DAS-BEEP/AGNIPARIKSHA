"""
AGNIPARIKSHA — Stage 8a: Decision Logic.

Aggregates the four distinct screening flags into a final verdict.
"""

from __future__ import annotations

import pandas as pd


def make_decision(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the final decision rule.
    
    A chip is FLAGGED if ANY of the following is true:
      1. pat_flag (univariate outlier)
      2. mv_flag (multivariate trajectory outlier)
      3. conformal_flag (95% upper bound on forecast > spec)
      4. lifetime_flag (projected life < mission requirement)
      
    Output columns:
      - verdict ("FLAG" or "PASS")
      - flag_reasons (list of strings naming which rules fired)
      - risk_score (int 0-4, number of rules fired)
    """
    df = df_wide.copy()
    
    rules = {
        "pat_flag": "PAT Univariate Outlier",
        "mv_flag": "Mahalanobis Trajectory Outlier",
        "conformal_flag": "Forecast 95% Upper Bound Exceeds Spec",
        "lifetime_flag": "Projected Lifetime < Mission Requirement"
    }
    
    # Ensure all columns exist
    for col in rules.keys():
        if col not in df.columns:
            # If a stage was skipped, default to False
            df[col] = False
            
    reasons_list = []
    score_list = []
    verdict_list = []
    
    for idx in df.index:
        chip_reasons = []
        score = 0
        
        for col, reason_text in rules.items():
            if df.loc[idx, col]:
                chip_reasons.append(reason_text)
                score += 1
                
        reasons_list.append(chip_reasons)
        score_list.append(score)
        verdict_list.append("FLAG" if score > 0 else "PASS")
        
    df["flag_reasons"] = reasons_list
    df["risk_score"] = score_list
    df["verdict"] = verdict_list
    
    return df
