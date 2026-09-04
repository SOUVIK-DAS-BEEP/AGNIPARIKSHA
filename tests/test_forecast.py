"""
Tests for src/forecast.py — Stage 5: ML Drift Forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from src.generator import generate_lot
from src.ingest import to_wide
from src.pat import pat_screen
from src.arrhenius import apply_arrhenius_fit
from src.forecast import build_features, train_forecaster, evaluate_forecaster_cv, apply_forecast
from src.constants import COL_LOT_ID, COL_I_LEAK


@pytest.fixture
def preprocessed_data():
    """Returns df_wide with pat and arrhenius applied."""
    # Ensure some chips fail PAT to test survivor-only logic
    df_long = generate_lot(n_chips=200, seed=42)
    df_wide = to_wide(df_long)
    df_pat = pat_screen(df_wide)
    return apply_arrhenius_fit(df_pat)


def test_build_features_columns(preprocessed_data: pd.DataFrame):
    """Ensure the feature matrix contains exactly the specified columns."""
    features = build_features(preprocessed_data)
    
    # 14 features specified in spec
    assert len(features.columns) == 14
    
    # Check a few specific ones
    assert "delta_i_leak_ua_0_24" in features.columns
    assert "lot_median_i_leak_0h" in features.columns
    assert "drift_rate_a_i_leak_ua" in features.columns
    

def test_build_features_lot_aggregates(preprocessed_data: pd.DataFrame):
    """Lot aggregates must be computed per lot using 0h data."""
    features = build_features(preprocessed_data)
    
    lot_groups = preprocessed_data.groupby(COL_LOT_ID)
    
    for lot_id, group in lot_groups:
        lot_idx = group.index
        # Compute manually
        manual_median = group[f"{COL_I_LEAK}_0h"].median()
        manual_std = group[f"{COL_I_LEAK}_0h"].std()
        
        # All chips in this lot should have these same aggregate values
        assert np.allclose(features.loc[lot_idx, "lot_median_i_leak_0h"], manual_median)
        # For single chip lots std might be NaN, our code fills with 0.0
        if len(group) == 1:
            assert np.allclose(features.loc[lot_idx, "lot_std_i_leak_0h"], 0.0)
        else:
            assert np.allclose(features.loc[lot_idx, "lot_std_i_leak_0h"], manual_std)


def test_train_forecaster():
    """Verify RF is default and SVR is supported."""
    # Dummy data
    X = pd.DataFrame(np.random.rand(10, 14))
    y = np.random.rand(10)
    
    rf_model = train_forecaster(X, y, model_type="rf")
    assert isinstance(rf_model, RandomForestRegressor)
    
    svr_model = train_forecaster(X, y, model_type="svr")
    assert isinstance(svr_model, SVR)


def test_evaluate_forecaster_cv_runs(preprocessed_data: pd.DataFrame):
    """Test that GroupKFold evaluation runs and returns metrics."""
    # Ensure at least 5 lots for 5-fold CV
    lots_data = []
    for i in range(5):
        dl = generate_lot(n_chips=20, lot_id=f"L00{i}", seed=42+i)
        lots_data.append(dl)
    df_multi_long = pd.concat(lots_data, ignore_index=True)
    df_multi_long["chip_id"] = df_multi_long["lot_id"] + "_" + df_multi_long["chip_id"]
    df_wide = to_wide(df_multi_long)
    df_pat = pat_screen(df_wide)
    df_arr = apply_arrhenius_fit(df_pat)
    
    metrics = evaluate_forecaster_cv(df_arr)
    
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] > 0


def test_apply_forecast_adds_column(preprocessed_data: pd.DataFrame):
    """apply_forecast should return df with pred_168h added."""
    df_res = apply_forecast(preprocessed_data)
    assert "pred_168h" in df_res.columns
    # Ensure no NaNs in predictions
    assert not df_res["pred_168h"].isna().any()
