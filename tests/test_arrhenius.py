"""
Tests for src/arrhenius.py — Stage 4: Arrhenius fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.arrhenius import acceleration_factor, fit_drift_rate, apply_arrhenius_fit
from src.generator import generate_lot
from src.ingest import to_wide


def test_acceleration_factor_07():
    """CRITICAL: AF(0.7 eV) must be ~938."""
    af = acceleration_factor(ea_ev=0.7)
    assert af == pytest.approx(938, abs=5), f"Expected ~938, got {af}"


def test_acceleration_factor_10():
    """
    CRITICAL: AF(1.0 eV) must be ~17,600.
    This assertion proves the 18x sensitivity to activation energy is understood.
    """
    af = acceleration_factor(ea_ev=1.0)
    assert af == pytest.approx(17600, abs=100), f"Expected ~17600, got {af}"


def test_fit_drift_rate_exact_values():
    """Verify log-linear extraction from 2 points."""
    t = np.array([0.0, 24.0])
    
    # p(t) = 10 * exp(0.01 * t)
    p0_true = 10.0
    a_true = 0.01
    
    v0 = p0_true
    v24 = p0_true * np.exp(a_true * 24.0)
    
    p0, a, r2, degraded = fit_drift_rate(t, np.array([v0, v24]))
    
    assert p0 == pytest.approx(p0_true)
    assert a == pytest.approx(a_true)
    assert r2 == 1.0
    assert not degraded


def test_fit_drift_rate_handles_non_positive():
    """Must not return NaN when values are <= 0. Should clamp to small positive."""
    t = np.array([0.0, 24.0])
    v = np.array([-5.0, 0.0])
    
    p0, a, r2, degraded = fit_drift_rate(t, v)
    
    assert not np.isnan(p0)
    assert not np.isnan(a)
    assert degraded is True


def test_fit_never_reads_96h_or_168h_columns():
    """
    CRITICAL: Stage 4 must only use 0h and 24h readings.
    We test this by deleting all 96h and 168h columns before calling the fit.
    If the fit tries to read them, it will throw a KeyError.
    """
    df_long = generate_lot(n_chips=10, seed=42)
    df_wide = to_wide(df_long)
    
    # Delete future knowledge
    cols_to_drop = [c for c in df_wide.columns if "96h" in c or "168h" in c]
    df_blind = df_wide.drop(columns=cols_to_drop)
    
    # This must run without errors
    df_fit = apply_arrhenius_fit(df_blind)
    
    # Verify outputs exist
    assert "drift_rate_a_i_leak_ua" in df_fit.columns
    assert "p0_fitted_i_leak_ua" in df_fit.columns
