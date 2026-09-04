"""
AGNIPARIKSHA — Stage 4: Arrhenius fit.

Physically-anchored drift modelling. Degradation under constant thermal
stress follows Arrhenius kinetics: p(t) = p0 * exp(a * t).

Key design constraints (SPEC.md Section 4, Stage 4):
  - Fit using ONLY the 0h and 24h readings. The entire value proposition is
    early prediction. Never let the fit see 96h or 168h values.
  - Closed-form log-linear least-squares fit: ln(p) = ln(p0) + a*t.
  - Handle non-positive values gracefully by clamping to a small epsilon
    rather than returning NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    COL_CHIP_ID,
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    BOLTZMANN_EV_PER_K,
)

# Parameters to fit
FIT_PARAMS: list[str] = [COL_I_LEAK, COL_V_TH, COL_T_DELAY]


def acceleration_factor(
    ea_ev: float = 0.7, 
    t_stress_c: float = 125.0, 
    t_use_c: float = 25.0
) -> float:
    """
    Compute the Arrhenius acceleration factor between stress and use temperatures.
    
    AF = exp[ (Ea / k) * (1/T_use - 1/T_stress) ]
    
    Parameters
    ----------
    ea_ev : float
        Activation energy in electronvolts (default 0.7).
    t_stress_c : float
        Stress temperature in Celsius (default 125.0).
    t_use_c : float
        Use (operating) temperature in Celsius (default 25.0).
        
    Returns
    -------
    float
        The acceleration factor multiplier.
    """
    t_stress_k = t_stress_c + 273.15
    t_use_k = t_use_c + 273.15
    
    # Exponent: (Ea / k) * (1/T_use - 1/T_stress)
    exponent = (ea_ev / BOLTZMANN_EV_PER_K) * (1.0 / t_use_k - 1.0 / t_stress_k)
    return float(np.exp(exponent))


def fit_drift_rate(t_hours: np.ndarray, values: np.ndarray) -> tuple[float, float, float, bool]:
    """
    Fit log-linear Arrhenius drift using ONLY the 0h and 24h readings.
    
    p(t) = p0 * exp(a * t)
    ln(p(t)) = ln(p0) + a * t
    
    Parameters
    ----------
    t_hours : np.ndarray
        Array of timepoints in hours. Must contain 0 and 24.
    values : np.ndarray
        Array of parameter values corresponding to t_hours.
        
    Returns
    -------
    tuple[float, float, float, bool]
        (p0, a, r_squared, fit_degraded)
    """
    # Extract ONLY t=0 and t=24
    idx_0 = np.where(t_hours == 0)[0]
    idx_24 = np.where(t_hours == 24)[0]
    
    if len(idx_0) == 0 or len(idx_24) == 0:
        raise ValueError("t_hours must contain both 0 and 24 to fit drift rate.")
        
    val_0 = float(values[idx_0[0]])
    val_24 = float(values[idx_24[0]])
    
    fit_degraded = False
    
    # Handle non-positive values by clamping to a small epsilon
    if val_0 <= 0 or val_24 <= 0:
        fit_degraded = True
        val_0 = max(val_0, 1e-9)
        val_24 = max(val_24, 1e-9)
        
    ln_p0 = np.log(val_0)
    ln_p24 = np.log(val_24)
    
    # Least squares with 2 points is exactly the line connecting them
    a = (ln_p24 - ln_p0) / 24.0
    p0 = val_0
    
    # R^2 for a 2-point fit is 1.0, unless slope is 0 (then technically undefined, we use 1.0)
    r_squared = 1.0
    
    return p0, a, r_squared, fit_degraded


def apply_arrhenius_fit(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Arrhenius fit to all parameters for each chip.
    
    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format data containing measurement columns for 0h and 24h.
        
    Returns
    -------
    pd.DataFrame
        Dataframe with added columns:
        - drift_rate_a_{param}
        - p0_fitted_{param}
        - fit_r2_{param}
        - fit_degraded_{param}
    """
    df = df_wide.copy()
    
    # We only use 0 and 24 for fitting
    t_arr = np.array([0.0, 24.0])
    
    for param in FIT_PARAMS:
        col_0h = f"{param}_0h"
        col_24h = f"{param}_24h"
        
        if col_0h not in df.columns or col_24h not in df.columns:
            raise ValueError(f"Missing required columns for fit: {col_0h} or {col_24h}")
            
        p0_list = []
        a_list = []
        r2_list = []
        degraded_list = []
        
        for idx in df.index:
            v0 = df.loc[idx, col_0h]
            v24 = df.loc[idx, col_24h]
            v_arr = np.array([v0, v24])
            
            p0, a, r2, degraded = fit_drift_rate(t_arr, v_arr)
            
            p0_list.append(p0)
            a_list.append(a)
            r2_list.append(r2)
            degraded_list.append(degraded)
            
        df[f"drift_rate_a_{param}"] = a_list
        df[f"p0_fitted_{param}"] = p0_list
        df[f"fit_r2_{param}"] = r2_list
        df[f"fit_degraded_{param}"] = degraded_list
        
    return df
