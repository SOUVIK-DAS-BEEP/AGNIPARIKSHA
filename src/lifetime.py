"""
AGNIPARIKSHA — Stage 6: Lifetime Projection.

Converts burn-in forecasts into mission-relevant lifetime estimates using
Arrhenius kinetics.

Key design constraints (SPEC.md Section 4, Stage 6):
  - AF is heavily sensitive to Activation Energy (Ea). Therefore, NEVER
    report a single absolute lifetime. Report a range (Ea=0.7 to 1.0).
  - Report `lifetime_rank_in_lot` which is Ea-independent and trustworthy.
  - Flag on the CONSERVATIVE (shorter) end of the range.
  - If a <= 0, cap at 50 years and flag `lifetime_capped = True`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    COL_I_LEAK,
    COL_LOT_ID,
    SPEC_LIMITS,
    MISSION_LIFE_YEARS,
    LIFETIME_CAP_YEARS,
    HOURS_PER_YEAR,
)
from src.arrhenius import acceleration_factor


def project_lifetime_years(
    p0: float, 
    drift_rate_a: float, 
    spec_limit: float, 
    af: float
) -> float:
    """
    Project lifetime in years at use conditions for a single Ea (via AF).
    Internal helper only — never surfaced directly to avoid false precision.
    
    Parameters
    ----------
    p0 : float
        Initial parameter value.
    drift_rate_a : float
        Drift rate constant 'a'.
    spec_limit : float
        The absolute spec limit.
    af : float
        Acceleration factor between stress and use temperature.
        
    Returns
    -------
    float
        Projected lifetime in years.
    """
    if drift_rate_a <= 0:
        return LIFETIME_CAP_YEARS
        
    if p0 >= spec_limit:
        return 0.0
        
    t_fail_stress = np.log(spec_limit / p0) / drift_rate_a
    t_fail_use = t_fail_stress * af
    
    years = t_fail_use / HOURS_PER_YEAR
    return min(years, LIFETIME_CAP_YEARS)


def project_lifetime_range(
    p0: float, 
    drift_rate_a: float, 
    spec_limit: float, 
    ea_low: float = 0.7, 
    ea_high: float = 1.0
) -> tuple[float, float, bool]:
    """
    Project the range of lifetimes across the plausible Activation Energy band.
    
    Returns
    -------
    tuple[float, float, bool]
        (years_at_ea_low, years_at_ea_high, is_capped)
    """
    af_low = acceleration_factor(ea_ev=ea_low)
    af_high = acceleration_factor(ea_ev=ea_high)
    
    y_low = project_lifetime_years(p0, drift_rate_a, spec_limit, af_low)
    y_high = project_lifetime_years(p0, drift_rate_a, spec_limit, af_high)
    
    # It's capped if either hits the cap, which happens if a <= 0 
    # or if the projected life is genuinely very long.
    is_capped = (drift_rate_a <= 0) or (y_high >= LIFETIME_CAP_YEARS)
    
    return y_low, y_high, is_capped


def lifetime_rank_in_lot(df_lot: pd.DataFrame, life_col: str = "life_years_ea07") -> pd.Series:
    """
    Compute Ea-independent lifetime rank within a lot.
    Rank = projected_life / lot_median(projected_life)
    
    Parameters
    ----------
    df_lot : pd.DataFrame
        Dataframe containing ONLY chips from a single lot.
    life_col : str
        Column to use for ranking. Since Ea is a scalar multiplier, 
        the ratio is identical whether ea07 or ea10 is used.
        
    Returns
    -------
    pd.Series
        Series of rank values.
    """
    med = df_lot[life_col].median()
    if med <= 0:
        # Prevent div by zero if an entire lot is already failed
        return pd.Series(1.0, index=df_lot.index)
    
    return df_lot[life_col] / med


def apply_lifetime_projection(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lifetime projections for leakage to all chips.
    
    Output columns: 
      - life_years_ea07
      - life_years_ea10
      - lifetime_capped
      - life_rank_in_lot
      - lifetime_flag
    """
    df = df_wide.copy()
    
    spec_max = SPEC_LIMITS[COL_I_LEAK]["max"]
    
    y_07_list = []
    y_10_list = []
    capped_list = []
    
    for idx in df.index:
        p0 = df.loc[idx, f"p0_fitted_{COL_I_LEAK}"]
        a = df.loc[idx, f"drift_rate_a_{COL_I_LEAK}"]
        
        y07, y10, capped = project_lifetime_range(p0, a, spec_max)
        
        y_07_list.append(y07)
        y_10_list.append(y10)
        capped_list.append(capped)
        
    df["life_years_ea07"] = y_07_list
    df["life_years_ea10"] = y_10_list
    df["lifetime_capped"] = capped_list
    
    # Flag on the CONSERVATIVE (shorter) end
    df["lifetime_flag"] = df["life_years_ea07"] < MISSION_LIFE_YEARS
    
    # Compute rank PER LOT
    df["life_rank_in_lot"] = np.nan
    for lot_id, lot_idx in df.groupby(COL_LOT_ID).groups.items():
        ranks = lifetime_rank_in_lot(df.loc[lot_idx], "life_years_ea07")
        df.loc[lot_idx, "life_rank_in_lot"] = ranks
        
    return df
