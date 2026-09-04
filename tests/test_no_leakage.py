"""
Tests for Data Leakage Prevention.

This is the most critical test in the project (SPEC.md Section 4, Stage 5).
It adversarially verifies that the forecasting model receives ZERO information
from the 96h or 168h measurements. 
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.generator import generate_lot
from src.ingest import to_wide
from src.pat import pat_screen
from src.arrhenius import apply_arrhenius_fit
from src.forecast import build_features


@pytest.fixture
def clean_data():
    """Returns df_wide with arrhenius fits applied."""
    df_long = generate_lot(n_chips=50, seed=42)
    df_wide = to_wide(df_long)
    df_pat = pat_screen(df_wide)
    return apply_arrhenius_fit(df_pat)


def test_no_leakage_in_column_names(clean_data: pd.DataFrame):
    """
    Assert that no feature column name explicitly mentions 96h or 168h.
    """
    features = build_features(clean_data)
    
    for col in features.columns:
        assert "96h" not in col.lower(), f"Leakage detected in feature name: {col}"
        assert "168h" not in col.lower(), f"Leakage detected in feature name: {col}"


def test_adversarial_poisoning(clean_data: pd.DataFrame):
    """
    CRITICAL: Adversarially test for implicit data leakage.
    
    We build features from the clean data. Then, we take the clean data,
    completely poison all 96h and 168h columns (set them to insane values),
    and build features again. 
    
    If ANY derivation path touches the future data, the poisoned feature
    matrix will differ from the clean one.
    """
    # 1. Build features from clean data
    clean_features = build_features(clean_data)
    
    # 2. Poison the future data in the input dataframe
    poisoned_data = clean_data.copy()
    
    for col in poisoned_data.columns:
        if "96h" in col or "168h" in col:
            # Inject insane values that would definitely alter any aggregate or calculation
            poisoned_data[col] = 999999999.9
            
    # 3. Build features from poisoned data
    poisoned_features = build_features(poisoned_data)
    
    # 4. Assert absolute identicality
    pd.testing.assert_frame_equal(
        clean_features, 
        poisoned_features, 
        check_exact=True,
        obj="Feature Matrix"
    )
    
    # Just to be absolutely certain, check that the values didn't both get poisoned somehow
    assert not (clean_features.values == 999999999.9).any()
