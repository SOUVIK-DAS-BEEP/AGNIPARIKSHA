"""
Tests for src/multivariate.py — Stage 3: Multivariate Outlier Screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from src.generator import generate_lot
from src.ingest import to_wide
from src.pat import pat_screen
from src.multivariate import compute_deltas, mahalanobis_screen
from src.constants import (
    COL_I_LEAK, COL_V_TH, COL_T_DELAY,
    CLASS_HEALTHY, CLASS_LATENT, CLASS_EARLY_FAIL
)


@pytest.fixture
def wide_data():
    """A generated lot in wide format, post-PAT."""
    df_long = generate_lot(n_chips=500, seed=42)
    df_wide = to_wide(df_long)
    return pat_screen(df_wide)


def test_compute_deltas(wide_data: pd.DataFrame):
    """Ensure delta features are computed correctly."""
    df_deltas, features = compute_deltas(wide_data)
    
    assert len(features) == 9
    assert f"delta_{COL_I_LEAK}_0_24" in features
    
    # Check a manual calculation
    expected_delta = wide_data.loc[0, f"{COL_I_LEAK}_24h"] - wide_data.loc[0, f"{COL_I_LEAK}_0h"]
    assert df_deltas.loc[0, f"delta_{COL_I_LEAK}_0_24"] == pytest.approx(expected_delta)


def test_chi2_threshold_applied(wide_data: pd.DataFrame):
    """Test that the threshold is exactly chi2(df=9, alpha=0.01)."""
    df_mv = mahalanobis_screen(wide_data)
    
    threshold = scipy.stats.chi2.ppf(0.99, df=9)
    
    flagged = df_mv[df_mv["mv_flag"]]
    unflagged = df_mv[~df_mv["mv_flag"]]
    
    if len(flagged) > 0:
        assert flagged["mahalanobis_d2"].min() > threshold
    if len(unflagged) > 0:
        assert unflagged["mahalanobis_d2"].max() <= threshold


def test_small_lot_fallback():
    """Guard: MinCovDet fails on tiny lots. Should fallback to diagonal without crashing."""
    df_long = generate_lot(n_chips=10, seed=42)  # 10 < 18 (2 * 9)
    df_wide = to_wide(df_long)
    
    df_mv = mahalanobis_screen(df_wide)
    
    # Should not crash and should set the low confidence flag
    assert df_mv["mv_low_confidence"].all()


def test_catches_latent_chips_missed_by_pat(wide_data: pd.DataFrame):
    """
    CRITICAL: The multivariate screen must catch latent chips that PAT missed.
    This is the core contribution of Stage 3.
    """
    df_mv = mahalanobis_screen(wide_data)
    
    latent = df_mv[df_mv["true_class"] == CLASS_LATENT]
    
    # Find latent chips missed by PAT
    missed_by_pat = latent[~latent["pat_flag"]]
    
    # Verify PAT missed a significant number
    assert len(missed_by_pat) > 10, "PAT caught too many latent chips for this test to be meaningful."
    
    # Check how many of these the MV screen catches
    caught_by_mv = missed_by_pat[missed_by_pat["mv_flag"]]
    
    catch_rate = len(caught_by_mv) / len(missed_by_pat)
    
    assert catch_rate > 0.50, (
        f"MV screen only caught {catch_rate:.0%} of the latent chips missed by PAT. "
        "It should catch most of them because they have correlated drift trajectories."
    )
