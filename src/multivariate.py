"""
AGNIPARIKSHA — Stage 3: Multivariate Outlier Screen.

PAT is univariate and provably misses joint drifts. This stage generalises
outlier detection to multiple dimensions using Mahalanobis distance.

Key design constraints (SPEC.md Section 4, Stage 3):
  - Features are drift DELTAS between consecutive timepoints, not raw values.
    Deltas capture trajectory shape, which PAT ignores.
  - MinCovDet is fitted PER LOT for robust covariance estimation.
  - Threshold is the theoretical chi-square critical value (alpha=0.01, df=9),
    not an empirical percentile.
  - Guard: If a lot has too few chips (< 2 * n_features), MinCovDet is
    unstable. Fall back to diagonal covariance (robust variance per feature)
    and set a mv_low_confidence flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
import scipy.stats

from src.constants import (
    COL_LOT_ID,
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    BURN_IN_HOURS,
)
from src.pat import robust_stats


def compute_deltas(df_wide: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute drift deltas between consecutive timepoints for all parameters.
    
    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        The original dataframe with added delta columns, and the list of
        the new feature column names (exactly 9 features).
    """
    df = df_wide.copy()
    features = []
    
    params = [COL_I_LEAK, COL_V_TH, COL_T_DELAY]
    intervals = [(0, 24), (24, 96), (96, 168)]
    
    for param in params:
        for t1, t2 in intervals:
            col_t1 = f"{param}_{t1}h"
            col_t2 = f"{param}_{t2}h"
            delta_col = f"delta_{param}_{t1}_{t2}"
            
            if col_t1 not in df.columns or col_t2 not in df.columns:
                raise ValueError(f"Missing required columns for delta: {col_t1} or {col_t2}")
                
            df[delta_col] = df[col_t2] - df[col_t1]
            features.append(delta_col)
            
    return df, features


def get_top_feature(diff: np.ndarray, precision: np.ndarray, feature_names: list[str]) -> str:
    """
    Identify the feature contributing most to the Mahalanobis distance.
    
    D^2 = diff^T * Precision * diff
    We approximate the contribution of feature i as: diff[i] * (Precision * diff)[i]
    """
    # Precision * diff is a vector. We multiply element-wise by diff.
    contributions = diff * np.dot(precision, diff)
    max_idx = np.argmax(np.abs(contributions))
    return feature_names[max_idx]


def mahalanobis_screen(df_wide: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """
    Multivariate outlier screen using robust Mahalanobis distance on drift trajectories.
    
    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format data containing measurement columns for 0, 24, 96, 168h.
    contamination : float
        Not strictly used for thresholding since we use the theoretical chi2 
        critical value, but can be passed to MinCovDet if desired (MinCovDet
        typically uses support fraction). We leave it as a parameter per spec.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added columns:
        - mahalanobis_d2 (float)
        - mv_flag (bool)
        - top_mv_feature (str)
        - mv_low_confidence (bool)
        plus the 9 delta feature columns.
    """
    df, feature_cols = compute_deltas(df_wide)
    n_features = len(feature_cols)
    
    # Theoretical threshold: chi-square critical value at alpha=0.01 (99th percentile)
    chi2_threshold = scipy.stats.chi2.ppf(0.99, df=n_features)
    
    # Initialize output columns
    df["mahalanobis_d2"] = np.nan
    df["mv_flag"] = False
    df["top_mv_feature"] = ""
    df["mv_low_confidence"] = False
    
    for lot_id, lot_idx in df.groupby(COL_LOT_ID).groups.items():
        lot_data = df.loc[lot_idx, feature_cols].values
        n_chips = len(lot_idx)
        
        # Guard: MinCovDet is unstable for small lots
        if n_chips < 2 * n_features:
            # Fallback to diagonal covariance
            df.loc[lot_idx, "mv_low_confidence"] = True
            
            location = np.zeros(n_features)
            variances = np.zeros(n_features)
            
            for j in range(n_features):
                med, r_sigma = robust_stats(lot_data[:, j])
                location[j] = med
                # Ensure variance is strictly positive to avoid division by zero
                variances[j] = max(r_sigma**2, 1e-12)
                
            precision = np.diag(1.0 / variances)
            
        else:
            # Fit MinCovDet
            mcd = MinCovDet(random_state=42)
            try:
                mcd.fit(lot_data)
                location = mcd.location_
                precision = mcd.precision_
            except ValueError:
                # If fitting fails (e.g., singular covariance), fallback
                df.loc[lot_idx, "mv_low_confidence"] = True
                location = np.zeros(n_features)
                variances = np.zeros(n_features)
                for j in range(n_features):
                    med, r_sigma = robust_stats(lot_data[:, j])
                    location[j] = med
                    variances[j] = max(r_sigma**2, 1e-12)
                precision = np.diag(1.0 / variances)
                
        # Compute squared Mahalanobis distance and top feature per chip
        for idx_pos, df_idx in enumerate(lot_idx):
            x = lot_data[idx_pos]
            diff = x - location
            d2 = np.dot(diff.T, np.dot(precision, diff))
            df.loc[df_idx, "mahalanobis_d2"] = d2
            
            if d2 > chi2_threshold:
                df.loc[df_idx, "mv_flag"] = True
                
            top_feat = get_top_feature(diff, precision, feature_cols)
            df.loc[df_idx, "top_mv_feature"] = top_feat
            
    return df
