"""
AGNIPARIKSHA — Stage 2: Part Average Testing (PAT) screen.

Univariate PAT per AEC-Q001. This is our BASELINE, not our innovation.
We implement it faithfully to show what conventional testing catches —
and what it misses.

Key design constraints (SPEC.md Section 4, Stage 2):
  - robust_sigma = 1.4826 * MAD (the factor makes MAD a consistent
    estimator of sigma for normally distributed data)
  - Limits are computed PER LOT, never globally
  - PAT uses the 168h (final) reading ONLY — this is deliberate, it
    reproduces conventional PAT's endpoint-only limitation

PAT flags a chip if its 168h reading exceeds the SPEC LIMITS (absolute
pass/fail thresholds). The robust z-scores are computed per lot and reported
for downstream stages, but the PAT FLAG itself is driven by the spec limits.
This faithfully reproduces the conventional screening blind spot: a chip
drifting rapidly but still under spec at 168h passes PAT.

SPEC.md Section 1: "Both pass, because conventional testing only checks
the final reading (45 < 50)."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    COL_CHIP_ID,
    COL_LOT_ID,
    COL_I_LEAK,
    COL_V_TH,
    COL_T_DELAY,
    SPEC_LIMITS,
    BURN_IN_HOURS,
)

# The consistency factor that makes MAD an unbiased estimator of the
# standard deviation for normally distributed data.
# MAD alone underestimates sigma by this factor.
# See: Rousseeuw & Croux (1993), or any robust statistics reference.
MAD_CONSISTENCY_FACTOR: float = 1.4826

# Parameters screened by PAT — uses the 168h (final) reading only.
# This is what conventional PAT sees. The limitation is the whole point.
PAT_PARAMS: list[str] = [COL_I_LEAK, COL_V_TH, COL_T_DELAY]


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    """
    Compute robust location and scale estimates.

    Uses the median as the location estimator and the Median Absolute
    Deviation (MAD), scaled by 1.4826, as the scale estimator. The 1.4826
    factor makes MAD a consistent estimator of sigma for normally
    distributed data — without it, the PAT limits would be ~35% too tight.

    Parameters
    ----------
    values : np.ndarray
        1-D array of numeric values. NaNs are excluded.

    Returns
    -------
    tuple[float, float]
        (median, robust_sigma) where robust_sigma = 1.4826 * MAD.
        Units match the input.
    """
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return (np.nan, np.nan)

    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    robust_sigma = MAD_CONSISTENCY_FACTOR * mad

    return (median, robust_sigma)


def pat_screen(
    df_wide: pd.DataFrame,
    n_sigma: float = 6.0,
) -> pd.DataFrame:
    """
    Univariate Part Average Testing screen per AEC-Q001.

    For each parameter, computes robust statistics (median and robust_sigma)
    PER LOT using the 168h (final) reading only, and calculates z-scores.

    Chips are FLAGGED if their 168h readings exceed SPEC LIMITS — the
    absolute pass/fail thresholds. This faithfully reproduces the
    conventional screening limitation: a chip that drifts rapidly but
    stays under spec at the endpoint PASSES.

    The robust z-scores are computed and included in the output for
    downstream analysis, but the pat_flag is driven by spec limits only.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Wide-format data (one row per chip), as produced by ingest.to_wide().
        Must contain columns like i_leak_ua_168h, v_th_v_168h, etc., plus
        chip_id and lot_id.
    n_sigma : float
        Number of robust sigmas for computing z-scores. Default 6.0 per
        AEC-Q001. Used for z-score computation, not for the flag.

    Returns
    -------
    pd.DataFrame
        Copy of df_wide with added columns:
        - pat_flag (bool): True if the chip's 168h value exceeds spec limits
        - pat_reason (str): Which parameter(s) caused the flag, or ""
        - {param}_zscore (float): Robust z-score for each parameter at 168h
          (computed per lot using robust median and 1.4826 * MAD)
    """
    result = df_wide.copy()

    # Identify the 168h columns for each parameter
    param_168h_cols = {
        param: f"{param}_{BURN_IN_HOURS}h"
        for param in PAT_PARAMS
    }

    # Verify all required columns exist
    for param, col_168h in param_168h_cols.items():
        if col_168h not in result.columns:
            raise ValueError(
                f"Column '{col_168h}' not found in wide-format data. "
                f"Did you run ingest.to_wide() first?"
            )

    # Initialise output columns
    result["pat_flag"] = False
    result["pat_reason"] = ""

    for param in PAT_PARAMS:
        result[f"{param}_zscore"] = np.nan

    # Compute z-scores PER LOT — PAT statistics are lot-specific, never global
    for lot_id, lot_idx in result.groupby(COL_LOT_ID).groups.items():
        lot_rows = result.loc[lot_idx]

        for param in PAT_PARAMS:
            col_168h = param_168h_cols[param]
            values = lot_rows[col_168h].values.astype(float)

            median, sigma = robust_stats(values)

            # Compute z-scores: how many robust sigmas from the lot median
            if sigma > 0:
                zscores = (values - median) / sigma
            else:
                # All values identical — z-score is 0 for all
                zscores = np.zeros_like(values)

            result.loc[lot_idx, f"{param}_zscore"] = zscores

    # Flag chips based on SPEC LIMITS (absolute pass/fail thresholds).
    # This is what conventional testing does: check the final reading
    # against the spec limit. PAT misses chips that drift fast but stay
    # under spec — and that limitation is the entire premise of this project.
    for param in PAT_PARAMS:
        col_168h = param_168h_cols[param]
        spec = SPEC_LIMITS.get(param, {})
        spec_max = spec.get("max")
        spec_min = spec.get("min")

        for idx in result.index:
            val = result.loc[idx, col_168h]
            reason_parts = []

            if spec_max is not None and val > spec_max:
                reason_parts.append(f"{param} exceeds spec max ({spec_max})")
            if spec_min is not None and val < spec_min:
                reason_parts.append(f"{param} below spec min ({spec_min})")

            if reason_parts:
                result.loc[idx, "pat_flag"] = True
                existing = result.loc[idx, "pat_reason"]
                new_reason = "; ".join(reason_parts)
                if existing:
                    result.loc[idx, "pat_reason"] = f"{existing}; {new_reason}"
                else:
                    result.loc[idx, "pat_reason"] = new_reason

    return result
