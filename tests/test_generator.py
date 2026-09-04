"""
Tests for src/generator.py

Validates the synthetic data generator per SPEC.md Section 5:
  1. No latent-defective chip exceeds 50 µA at 168 h
  2. DEMO-CHIP-A and DEMO-CHIP-B are present with exactly the specified values
  3. The generator is reproducible for a fixed seed
  4. The three populations appear in roughly the specified proportions
"""

import pytest
import numpy as np
import pandas as pd

from src.generator import generate_lot, generate_lot_from_preset
from src.constants import (
    SPEC_LIMITS,
    TIMEPOINTS,
    STRESS_TEMP_C,
    NOMINAL_VOLTAGE_V,
    DEMO_CHIP_A_ID,
    DEMO_CHIP_B_ID,
    DEMO_CHIP_A_LEAKAGE,
    DEMO_CHIP_B_LEAKAGE,
    CLASS_HEALTHY,
    CLASS_LATENT,
    CLASS_EARLY_FAIL,
    BURN_IN_HOURS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_lot() -> pd.DataFrame:
    """Generate a default lot (500 chips, seed=42) for testing."""
    return generate_lot(n_chips=500, lot_id="L001", seed=42)


@pytest.fixture
def small_lot() -> pd.DataFrame:
    """Generate a small lot for fast tests."""
    return generate_lot(n_chips=100, lot_id="L001", seed=42)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchema:
    """Verify the output DataFrame has the correct schema."""

    def test_required_columns(self, default_lot: pd.DataFrame):
        expected_cols = {
            "chip_id", "lot_id", "wafer_id", "t_hours",
            "i_leak_ua", "v_th_v", "t_delay_ns",
            "temp_c", "voltage_v", "true_class",
        }
        assert expected_cols == set(default_lot.columns)

    def test_four_rows_per_chip(self, default_lot: pd.DataFrame):
        """Each chip must have exactly 4 rows (one per timepoint)."""
        rows_per_chip = default_lot.groupby("chip_id").size()
        assert (rows_per_chip == 4).all(), (
            f"Some chips don't have exactly 4 rows: "
            f"{rows_per_chip[rows_per_chip != 4].to_dict()}"
        )

    def test_correct_timepoints(self, default_lot: pd.DataFrame):
        """Each chip must have measurements at t=0, 24, 96, 168."""
        for chip_id, group in default_lot.groupby("chip_id"):
            assert sorted(group["t_hours"].tolist()) == TIMEPOINTS, (
                f"Chip {chip_id} has wrong timepoints: {group['t_hours'].tolist()}"
            )

    def test_no_duplicate_chip_timepoint_pairs(self, default_lot: pd.DataFrame):
        """No (chip_id, t_hours) pair should appear twice."""
        dupes = default_lot.duplicated(subset=["chip_id", "t_hours"])
        assert not dupes.any(), "Found duplicate (chip_id, t_hours) pairs"

    def test_numeric_columns_are_numeric(self, default_lot: pd.DataFrame):
        for col in ["i_leak_ua", "v_th_v", "t_delay_ns", "temp_c", "voltage_v"]:
            assert pd.api.types.is_numeric_dtype(default_lot[col]), (
                f"Column '{col}' is not numeric"
            )

    def test_stress_temperature(self, default_lot: pd.DataFrame):
        """All readings should be at the stress temperature."""
        assert (default_lot["temp_c"] == STRESS_TEMP_C).all()

    def test_stress_voltage(self, default_lot: pd.DataFrame):
        """All readings should be at the nominal voltage."""
        assert (default_lot["voltage_v"] == NOMINAL_VOLTAGE_V).all()

    def test_chip_count(self, default_lot: pd.DataFrame):
        """Should have 500 random chips + 2 demo chips = 502."""
        n_unique = default_lot["chip_id"].nunique()
        assert n_unique == 502, f"Expected 502 unique chips, got {n_unique}"


# ---------------------------------------------------------------------------
# CRITICAL: No latent chip exceeds spec at 168 h
# ---------------------------------------------------------------------------

class TestLatentSpecConstraint:
    """
    THE most important test: no latent-defective chip exceeds the leakage
    spec at 168 h. This constraint is the entire point of the dataset.

    If a latent chip exceeds spec, it would be caught by conventional
    testing and is NOT the population we care about. The resample loop
    must guarantee this.
    """

    def test_no_latent_chip_exceeds_spec_at_168h(self, default_lot: pd.DataFrame):
        """No latent chip may have i_leak_ua >= 50.0 µA at t=168 h."""
        latent_at_168 = default_lot[
            (default_lot["true_class"] == CLASS_LATENT) &
            (default_lot["t_hours"] == BURN_IN_HOURS)
        ]
        spec_max = SPEC_LIMITS["i_leak_ua"]["max"]
        violations = latent_at_168[latent_at_168["i_leak_ua"] >= spec_max]
        assert len(violations) == 0, (
            f"{len(violations)} latent chips exceed {spec_max} µA at 168h:\n"
            f"{violations[['chip_id', 'i_leak_ua']].to_string()}"
        )

    def test_latent_chips_show_elevated_drift(self, default_lot: pd.DataFrame):
        """Latent chips should show clearly more drift than healthy chips."""
        latent = default_lot[default_lot["true_class"] == CLASS_LATENT]
        healthy = default_lot[default_lot["true_class"] == CLASS_HEALTHY]

        # Compute mean drift (168h - 0h) for each population
        def mean_drift(df: pd.DataFrame) -> float:
            at_0 = df[df["t_hours"] == 0]["i_leak_ua"].mean()
            at_168 = df[df["t_hours"] == 168]["i_leak_ua"].mean()
            return at_168 - at_0

        latent_drift = mean_drift(latent)
        healthy_drift = mean_drift(healthy)

        assert latent_drift > healthy_drift * 5, (
            f"Latent drift ({latent_drift:.2f} µA) should be much larger "
            f"than healthy drift ({healthy_drift:.2f} µA)"
        )

    def test_multiple_seeds_no_violations(self):
        """Check the constraint holds across several seeds."""
        spec_max = SPEC_LIMITS["i_leak_ua"]["max"]
        for seed in [1, 7, 42, 99, 123, 456, 789]:
            df = generate_lot(n_chips=200, seed=seed)
            latent_168 = df[
                (df["true_class"] == CLASS_LATENT) &
                (df["t_hours"] == BURN_IN_HOURS)
            ]
            violations = latent_168[latent_168["i_leak_ua"] >= spec_max]
            assert len(violations) == 0, (
                f"Seed {seed}: {len(violations)} latent chips exceed spec"
            )


# ---------------------------------------------------------------------------
# DEMO-CHIP-A and DEMO-CHIP-B
# ---------------------------------------------------------------------------

class TestDemoChips:
    """
    DEMO-CHIP-A and DEMO-CHIP-B must be present with exactly the leakage
    values specified in SPEC.md Section 1.
    """

    def test_demo_chip_a_present(self, default_lot: pd.DataFrame):
        assert DEMO_CHIP_A_ID in default_lot["chip_id"].values

    def test_demo_chip_b_present(self, default_lot: pd.DataFrame):
        assert DEMO_CHIP_B_ID in default_lot["chip_id"].values

    def test_chip_a_exact_leakage_values(self, default_lot: pd.DataFrame):
        """Chip A leakage: 10.0, 10.1, 10.2, 10.3 µA — no noise applied."""
        chip_a = default_lot[default_lot["chip_id"] == DEMO_CHIP_A_ID].sort_values("t_hours")
        for _, row in chip_a.iterrows():
            tp = row["t_hours"]
            expected = DEMO_CHIP_A_LEAKAGE[tp]
            assert row["i_leak_ua"] == pytest.approx(expected), (
                f"DEMO-CHIP-A at t={tp}h: expected {expected} µA, "
                f"got {row['i_leak_ua']:.4f} µA"
            )

    def test_chip_b_exact_leakage_values(self, default_lot: pd.DataFrame):
        """Chip B leakage: 12.0, 22.0, 35.0, 45.0 µA — no noise applied."""
        chip_b = default_lot[default_lot["chip_id"] == DEMO_CHIP_B_ID].sort_values("t_hours")
        for _, row in chip_b.iterrows():
            tp = row["t_hours"]
            expected = DEMO_CHIP_B_LEAKAGE[tp]
            assert row["i_leak_ua"] == pytest.approx(expected), (
                f"DEMO-CHIP-B at t={tp}h: expected {expected} µA, "
                f"got {row['i_leak_ua']:.4f} µA"
            )

    def test_chip_a_classified_healthy(self, default_lot: pd.DataFrame):
        chip_a = default_lot[default_lot["chip_id"] == DEMO_CHIP_A_ID]
        assert (chip_a["true_class"] == CLASS_HEALTHY).all()

    def test_chip_b_classified_latent(self, default_lot: pd.DataFrame):
        chip_b = default_lot[default_lot["chip_id"] == DEMO_CHIP_B_ID]
        assert (chip_b["true_class"] == CLASS_LATENT).all()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """The generator must produce identical output for a fixed seed."""

    def test_same_seed_same_output(self):
        df1 = generate_lot(n_chips=100, seed=42)
        df2 = generate_lot(n_chips=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_output(self):
        df1 = generate_lot(n_chips=100, seed=42)
        df2 = generate_lot(n_chips=100, seed=99)
        # Same shape, different values
        assert df1.shape == df2.shape
        assert not df1["i_leak_ua"].equals(df2["i_leak_ua"])


# ---------------------------------------------------------------------------
# Population proportions
# ---------------------------------------------------------------------------

class TestPopulationProportions:
    """The three populations should appear in roughly the specified proportions."""

    def test_default_proportions(self, default_lot: pd.DataFrame):
        """Default: 85% healthy, 10% latent, 5% early_fail (± tolerance)."""
        # Get unique chips (not counting demo chips)
        chips = default_lot[
            ~default_lot["chip_id"].isin([DEMO_CHIP_A_ID, DEMO_CHIP_B_ID])
        ]
        unique_chips = chips.drop_duplicates(subset=["chip_id"])
        n_total = len(unique_chips)

        n_healthy = (unique_chips["true_class"] == CLASS_HEALTHY).sum()
        n_latent = (unique_chips["true_class"] == CLASS_LATENT).sum()
        n_early = (unique_chips["true_class"] == CLASS_EARLY_FAIL).sum()

        # Allow ±5% tolerance for rounding
        assert n_healthy / n_total == pytest.approx(0.85, abs=0.05), (
            f"Healthy fraction: {n_healthy / n_total:.2%}, expected ~85%"
        )
        assert n_latent / n_total == pytest.approx(0.10, abs=0.05), (
            f"Latent fraction: {n_latent / n_total:.2%}, expected ~10%"
        )
        assert n_early / n_total == pytest.approx(0.05, abs=0.05), (
            f"Early-fail fraction: {n_early / n_total:.2%}, expected ~5%"
        )

    def test_all_three_classes_present(self, default_lot: pd.DataFrame):
        classes = set(default_lot["true_class"].unique())
        expected = {CLASS_HEALTHY, CLASS_LATENT, CLASS_EARLY_FAIL}
        assert classes == expected

    def test_custom_proportions(self):
        """Custom fractions should be respected."""
        df = generate_lot(n_chips=200, seed=42,
                          healthy_frac=0.70, latent_frac=0.20, early_fail_frac=0.10)
        chips = df[~df["chip_id"].isin([DEMO_CHIP_A_ID, DEMO_CHIP_B_ID])]
        unique_chips = chips.drop_duplicates(subset=["chip_id"])
        n_total = len(unique_chips)

        n_healthy = (unique_chips["true_class"] == CLASS_HEALTHY).sum()
        n_latent = (unique_chips["true_class"] == CLASS_LATENT).sum()

        assert n_healthy / n_total == pytest.approx(0.70, abs=0.05)
        assert n_latent / n_total == pytest.approx(0.20, abs=0.05)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    """Test the preset-based generator."""

    def test_realistic_mixed_preset(self):
        df = generate_lot_from_preset("realistic_mixed", n_chips=100, seed=42)
        assert len(df) > 0
        assert df["chip_id"].nunique() == 102  # 100 + 2 demo

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            generate_lot_from_preset("nonexistent")

    def test_all_presets_generate_valid_data(self):
        for preset in ["healthy_heavy", "defect_heavy", "realistic_mixed"]:
            df = generate_lot_from_preset(preset, n_chips=50, seed=42)
            assert df["chip_id"].nunique() == 52  # 50 + 2 demo
            # Check no latent exceeds spec
            latent_168 = df[
                (df["true_class"] == CLASS_LATENT) &
                (df["t_hours"] == BURN_IN_HOURS)
            ]
            spec_max = SPEC_LIMITS["i_leak_ua"]["max"]
            assert (latent_168["i_leak_ua"] < spec_max).all()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Test that invalid inputs are rejected clearly."""

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            generate_lot(healthy_frac=0.5, latent_frac=0.3, early_fail_frac=0.3)

    def test_zero_chips_with_invalid_fractions_raises(self):
        """All-zero fractions don't sum to 1.0, so this should raise."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            generate_lot(n_chips=0, seed=42,
                          healthy_frac=0.0, latent_frac=0.0, early_fail_frac=0.0)


# ---------------------------------------------------------------------------
# Physical plausibility
# ---------------------------------------------------------------------------

class TestPhysicalPlausibility:
    """Sanity checks on the generated data."""

    def test_leakage_values_positive(self, default_lot: pd.DataFrame):
        """Leakage current must be positive (physics)."""
        assert (default_lot["i_leak_ua"] > 0).all()

    def test_threshold_voltage_reasonable(self, default_lot: pd.DataFrame):
        """v_th should be in a physically reasonable range."""
        assert default_lot["v_th_v"].min() > 0.1   # volts
        assert default_lot["v_th_v"].max() < 1.0   # volts

    def test_delay_positive(self, default_lot: pd.DataFrame):
        """Propagation delay must be positive."""
        assert (default_lot["t_delay_ns"] > 0).all()

    def test_early_fail_chips_exceed_spec(self, default_lot: pd.DataFrame):
        """
        Early-failure chips should mostly cross the spec limit during burn-in.
        (They die during burn-in — that's how they're defined.)
        """
        early_168 = default_lot[
            (default_lot["true_class"] == CLASS_EARLY_FAIL) &
            (default_lot["t_hours"] == BURN_IN_HOURS)
        ]
        spec_max = SPEC_LIMITS["i_leak_ua"]["max"]
        exceed_frac = (early_168["i_leak_ua"] >= spec_max).mean()
        # Most early-fail chips should exceed spec, but some may not
        # due to their initial value and drift rate distribution
        assert exceed_frac > 0.5, (
            f"Only {exceed_frac:.0%} of early-fail chips exceed spec at 168h, "
            f"expected most to cross the limit"
        )

    def test_healthy_chips_well_under_spec(self, default_lot: pd.DataFrame):
        """Healthy chips should be well below the spec limit."""
        healthy_168 = default_lot[
            (default_lot["true_class"] == CLASS_HEALTHY) &
            (default_lot["t_hours"] == BURN_IN_HOURS)
        ]
        spec_max = SPEC_LIMITS["i_leak_ua"]["max"]
        assert (healthy_168["i_leak_ua"] < spec_max * 0.5).all(), (
            "Some healthy chips are too close to the spec limit"
        )
