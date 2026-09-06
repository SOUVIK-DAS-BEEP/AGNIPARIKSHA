"""
Tests for src/lifetime.py — Stage 6: Lifetime Projection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.generator import generate_lot
from src.ingest import to_wide
from src.pat import pat_screen
from src.arrhenius import apply_arrhenius_fit, acceleration_factor
from src.lifetime import (
    project_lifetime_years,
    project_lifetime_range,
    lifetime_rank_in_lot,
    apply_lifetime_projection,
)
from src.constants import COL_LOT_ID, MISSION_LIFE_YEARS, LIFETIME_CAP_YEARS


@pytest.fixture
def test_data():
    """Returns df_wide with arrhenius fits applied."""
    df_long = generate_lot(n_chips=100, seed=42)
    df_wide = to_wide(df_long)
    df_pat = pat_screen(df_wide)
    return apply_arrhenius_fit(df_pat)


def test_project_lifetime_years():
    """Check basic lifetime projection math."""
    p0 = 10.0
    spec_limit = 50.0
    a = 0.01  # positive drift
    af = 940.0
    
    # t_stress = ln(50/10) / 0.01 = 160.94 hours
    # t_use = 160.94 * 940 = 151287 hours
    # years = 151287 / (24 * 365.25) = 17.25
    expected_years = (np.log(spec_limit / p0) / a) * af / (24 * 365.25)
    
    years = project_lifetime_years(p0, a, spec_limit, af)
    assert years == pytest.approx(expected_years)


def test_project_lifetime_capped():
    """Negative drift rate should return capped lifetime."""
    years = project_lifetime_years(p0=10.0, drift_rate_a=-0.01, spec_limit=50.0, af=940.0)
    assert years == LIFETIME_CAP_YEARS


def test_project_lifetime_already_failed():
    """If p0 >= spec limit, lifetime should be 0."""
    years = project_lifetime_years(p0=60.0, drift_rate_a=0.01, spec_limit=50.0, af=940.0)
    assert years == 0.0


def test_project_lifetime_range():
    """Check that range returns correctly ordered tuple and cap flag."""
    p0 = 10.0
    spec_limit = 50.0
    a = 0.1
    
    y07, y10, capped = project_lifetime_range(p0, a, spec_limit, ea_low=0.7, ea_high=1.0)
    
    assert y10 > y07
    assert not capped
    
    # Check cap logic
    _, _, capped2 = project_lifetime_range(10.0, -0.01, 50.0)
    assert capped2
    

def test_lifetime_rank_in_lot():
    """Rank should be 1.0 for the median chip."""
    df = pd.DataFrame({"life_years": [10.0, 20.0, 30.0, 40.0, 50.0]})
    # Median is 30.0
    ranks = lifetime_rank_in_lot(df, "life_years")
    assert ranks.iloc[2] == 1.0
    assert ranks.iloc[0] == 10.0 / 30.0


def test_apply_lifetime_projection(test_data: pd.DataFrame):
    """Verify output columns of apply_lifetime_projection."""
    df_life = apply_lifetime_projection(test_data)
    
    # Assert correct columns
    assert "life_years_ea07" in df_life.columns
    assert "life_years_ea10" in df_life.columns
    assert "lifetime_capped" in df_life.columns
    assert "life_rank_in_lot" in df_life.columns
    assert "lifetime_flag" in df_life.columns
    
    # CRITICAL: NO scalar projected_life_years
    assert "projected_life_years" not in df_life.columns
    
    # Flag should fire on the optimistic (ea10) end — only reject when
    # even the best-case Ea projection is below mission life.
    expected_flags = df_life["life_years_ea10"] < MISSION_LIFE_YEARS
    assert (df_life["lifetime_flag"] == expected_flags).all()
