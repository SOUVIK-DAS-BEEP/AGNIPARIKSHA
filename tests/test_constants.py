"""
Tests for src/constants.py

Validates that all constants from SPEC.md Section 3.2 are present, correctly
typed, and internally consistent.
"""

import pytest

from src.constants import (
    SPEC_LIMITS,
    BURN_IN_HOURS,
    TIMEPOINTS,
    STRESS_TEMP_C,
    USE_TEMP_C,
    STRESS_TEMP_K,
    USE_TEMP_K,
    DEFAULT_EA_EV,
    BOLTZMANN_EV_PER_K,
    MISSION_LIFE_YEARS,
    NOMINAL_VOLTAGE_V,
    DEMO_CHIP_A_ID,
    DEMO_CHIP_B_ID,
    DEMO_CHIP_A_LEAKAGE,
    DEMO_CHIP_B_LEAKAGE,
    CLASS_HEALTHY,
    CLASS_LATENT,
    CLASS_EARLY_FAIL,
    ASSUMED_DEFECT_CORRELATION,
    ASSUMED_HEALTHY_CORRELATION,
    MEASUREMENT_NOISE_FRAC,
    GLITCH_PROBABILITY,
    GLITCH_MAGNITUDE_FRAC,
    PRESETS,
    DEFAULT_HEALTHY_FRAC,
    DEFAULT_LATENT_FRAC,
    DEFAULT_EARLY_FAIL_FRAC,
    LIFETIME_CAP_YEARS,
    HOURS_PER_YEAR,
)


class TestSpecLimits:
    """Tests for SPEC_LIMITS dict — SPEC.md Section 3.2."""

    def test_all_parameters_present(self):
        assert "i_leak_ua" in SPEC_LIMITS
        assert "v_th_v" in SPEC_LIMITS
        assert "t_delay_ns" in SPEC_LIMITS

    def test_leakage_limit(self):
        assert SPEC_LIMITS["i_leak_ua"]["max"] == 50.0
        assert SPEC_LIMITS["i_leak_ua"]["min"] is None

    def test_threshold_voltage_limits(self):
        assert SPEC_LIMITS["v_th_v"]["max"] == 0.55
        assert SPEC_LIMITS["v_th_v"]["min"] == 0.35

    def test_delay_limit(self):
        assert SPEC_LIMITS["t_delay_ns"]["max"] == 2.50
        assert SPEC_LIMITS["t_delay_ns"]["min"] is None


class TestBurnInParameters:
    """Tests for burn-in timing and temperature constants."""

    def test_burn_in_hours(self):
        assert BURN_IN_HOURS == 168

    def test_timepoints(self):
        assert TIMEPOINTS == [0, 24, 96, 168]
        assert len(TIMEPOINTS) == 4

    def test_stress_temperature(self):
        assert STRESS_TEMP_C == 125.0

    def test_use_temperature(self):
        assert USE_TEMP_C == 25.0

    def test_kelvin_conversions(self):
        """Verify K = °C + 273.15."""
        assert STRESS_TEMP_K == pytest.approx(398.15)
        assert USE_TEMP_K == pytest.approx(298.15)


class TestPhysicsConstants:
    """Tests for physics constants."""

    def test_default_activation_energy(self):
        assert DEFAULT_EA_EV == 0.7

    def test_boltzmann_constant(self):
        assert BOLTZMANN_EV_PER_K == 8.617e-5


class TestMissionParameters:
    """Tests for mission parameters."""

    def test_mission_life(self):
        assert MISSION_LIFE_YEARS == 15.0

    def test_nominal_voltage(self):
        assert NOMINAL_VOLTAGE_V == 1.2


class TestDemoChips:
    """Tests for DEMO-CHIP-A and DEMO-CHIP-B — SPEC.md Section 1."""

    def test_demo_chip_ids(self):
        assert DEMO_CHIP_A_ID == "DEMO-CHIP-A"
        assert DEMO_CHIP_B_ID == "DEMO-CHIP-B"

    def test_chip_a_leakage_values(self):
        """Chip A: 10.0, 10.1, 10.2, 10.3 µA — nearly flat."""
        assert DEMO_CHIP_A_LEAKAGE == {0: 10.0, 24: 10.1, 96: 10.2, 168: 10.3}

    def test_chip_b_leakage_values(self):
        """Chip B: 12.0, 22.0, 35.0, 45.0 µA — rapidly drifting but under spec."""
        assert DEMO_CHIP_B_LEAKAGE == {0: 12.0, 24: 22.0, 96: 35.0, 168: 45.0}

    def test_chip_a_all_under_spec(self):
        """Chip A must be under 50 µA at all timepoints."""
        for tp, val in DEMO_CHIP_A_LEAKAGE.items():
            assert val < SPEC_LIMITS["i_leak_ua"]["max"], (
                f"Chip A exceeds spec at t={tp}h: {val} µA"
            )

    def test_chip_b_all_under_spec(self):
        """Chip B must be under 50 µA at all timepoints (it passes conventional testing)."""
        for tp, val in DEMO_CHIP_B_LEAKAGE.items():
            assert val < SPEC_LIMITS["i_leak_ua"]["max"], (
                f"Chip B exceeds spec at t={tp}h: {val} µA"
            )

    def test_chip_b_shows_drift(self):
        """Chip B must show clearly elevated drift (quadrupled in a week)."""
        ratio = DEMO_CHIP_B_LEAKAGE[168] / DEMO_CHIP_B_LEAKAGE[0]
        assert ratio > 3.0, (
            f"Chip B leakage ratio 168h/0h = {ratio:.1f}, expected > 3.0"
        )


class TestPopulationFractions:
    """Tests for default population fractions."""

    def test_fractions_sum_to_one(self):
        total = DEFAULT_HEALTHY_FRAC + DEFAULT_LATENT_FRAC + DEFAULT_EARLY_FAIL_FRAC
        assert total == pytest.approx(1.0)

    def test_default_values(self):
        assert DEFAULT_HEALTHY_FRAC == 0.85
        assert DEFAULT_LATENT_FRAC == 0.10
        assert DEFAULT_EARLY_FAIL_FRAC == 0.05


class TestCorrelationConstants:
    """Tests for correlation constants — honestly labelled assumptions."""

    def test_defect_correlation(self):
        assert ASSUMED_DEFECT_CORRELATION == 0.6

    def test_healthy_correlation(self):
        assert ASSUMED_HEALTHY_CORRELATION == 0.1

    def test_defect_higher_than_healthy(self):
        """Defective chips should have higher parameter correlation."""
        assert ASSUMED_DEFECT_CORRELATION > ASSUMED_HEALTHY_CORRELATION


class TestPresets:
    """Tests for generator presets."""

    def test_preset_names(self):
        assert "healthy_heavy" in PRESETS
        assert "defect_heavy" in PRESETS
        assert "realistic_mixed" in PRESETS

    def test_presets_sum_to_one(self):
        for name, (h, l, e) in PRESETS.items():
            assert h + l + e == pytest.approx(1.0), (
                f"Preset '{name}' fractions sum to {h + l + e}, expected 1.0"
            )


class TestNoiseConstants:
    """Tests for measurement noise parameters."""

    def test_noise_fraction(self):
        assert MEASUREMENT_NOISE_FRAC == 0.02

    def test_glitch_probability(self):
        assert GLITCH_PROBABILITY == 0.005

    def test_glitch_magnitude(self):
        assert GLITCH_MAGNITUDE_FRAC == 0.15


class TestLifetimeConstants:
    """Tests for lifetime projection constants."""

    def test_lifetime_cap(self):
        assert LIFETIME_CAP_YEARS == 50.0

    def test_hours_per_year(self):
        assert HOURS_PER_YEAR == pytest.approx(24.0 * 365.25)
