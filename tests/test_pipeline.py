"""
Tests for src/pipeline.py — End-to-end integration test.
"""

from __future__ import annotations

import pandas as pd
from src.generator import generate_lot
from src.pipeline import run_pipeline


def test_pipeline_end_to_end():
    """Smoke test ensuring the entire pipeline runs without crashing."""
    df_long = generate_lot(n_chips=50, seed=42)
    
    df, model, report = run_pipeline(df_long)
    
    # Check that final decision columns exist
    assert "verdict" in df.columns
    assert "flag_reasons" in df.columns
    assert "risk_score" in df.columns
    
    # Check that model was returned
    assert model is not None
    
    # Check that evaluation ran
    assert report is not None
    assert report.full_recall_latent >= 0.0
    assert report.conformal_coverage >= 0.0
