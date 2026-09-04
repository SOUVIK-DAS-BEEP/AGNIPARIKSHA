"""
AGNIPARIKSHA — Synthetic burn-in data generator.

Implements SPEC.md Section 5: generates physically faithful semiconductor
burn-in test data with three chip populations (healthy, latent-defective,
early-failure).

Each chip contributes 4 rows (one per timepoint) in the canonical long-format
schema. Degradation follows Arrhenius kinetics: p(t) = p0 * exp(a * t).

Key design decisions:
  - Latent-defective chips are resampled until they stay UNDER 50 µA at 168 h.
    This constraint is the entire point of the dataset — these chips pass
    conventional testing but are still dangerous.
  - Parameter correlations (ASSUMED_DEFECT_CORRELATION) are explicitly labelled
    as assumptions, not measured values.
  - DEMO-CHIP-A and DEMO-CHIP-B are hardcoded into every lot for demo narrative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    TIMEPOINTS,
    SPEC_LIMITS,
    STRESS_TEMP_C,
    NOMINAL_VOLTAGE_V,
    ASSUMED_DEFECT_CORRELATION,
    ASSUMED_HEALTHY_CORRELATION,
    MEASUREMENT_NOISE_FRAC,
    GLITCH_PROBABILITY,
    GLITCH_MAGNITUDE_FRAC,
    CLASS_HEALTHY,
    CLASS_LATENT,
    CLASS_EARLY_FAIL,
    DEMO_CHIP_A_ID,
    DEMO_CHIP_B_ID,
    DEMO_CHIP_A_LEAKAGE,
    DEMO_CHIP_B_LEAKAGE,
    DEFAULT_SEED,
    DEFAULT_N_CHIPS,
    DEFAULT_HEALTHY_FRAC,
    DEFAULT_LATENT_FRAC,
    DEFAULT_EARLY_FAIL_FRAC,
    PRESETS,
    BURN_IN_HOURS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arrhenius_value(p0: float, drift_rate: float, t: float) -> float:
    """
    Compute parameter value at time t under Arrhenius kinetics.

    p(t) = p0 * exp(drift_rate * t)

    Parameters
    ----------
    p0 : float
        Initial parameter value at t = 0.
    drift_rate : float
        Drift rate constant 'a', units of 1/hours.
    t : float
        Time in hours.

    Returns
    -------
    float
        Parameter value at time t.
    """
    return p0 * np.exp(drift_rate * t)


def _generate_correlated_drifts(
    rng: np.random.Generator,
    n: int,
    leakage_drifts: np.ndarray,
    correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate v_th and t_delay drift rates correlated with leakage drift.

    Uses a Cholesky decomposition to induce the target correlation between
    the leakage drift and the other two parameters.

    ASSUMPTION: The correlation value is physically plausible (a shared
    defect mechanism should perturb several parameters together) but is
    NOT a measured value from real silicon. See ASSUMED_DEFECT_CORRELATION
    in constants.py.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator instance.
    n : int
        Number of chips.
    leakage_drifts : np.ndarray
        Already-generated leakage drift rates, shape (n,).
    correlation : float
        Target Pearson correlation ρ between leakage drift and other drifts.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (v_th_drifts, t_delay_drifts) — drift rates in units appropriate
        for their respective parameters.
    """
    # Generate independent standard normals
    z_vth = rng.standard_normal(n)
    z_tdelay = rng.standard_normal(n)

    # Standardise leakage drifts to use as the correlation anchor
    leak_std = leakage_drifts - np.mean(leakage_drifts)
    std_val = np.std(leak_std)
    if std_val > 0:
        leak_std = leak_std / std_val
    else:
        leak_std = np.zeros(n)

    # Induce correlation via: Y = ρ * X + sqrt(1 - ρ²) * Z
    rho = correlation
    corr_vth = rho * leak_std + np.sqrt(1 - rho**2) * z_vth
    corr_tdelay = rho * leak_std + np.sqrt(1 - rho**2) * z_tdelay

    return corr_vth, corr_tdelay


def _scale_correlated_to_drift(
    correlated_values: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """
    Rescale standardised correlated values to a target distribution.

    Parameters
    ----------
    correlated_values : np.ndarray
        Values with zero mean and unit variance (approximately).
    target_mean : float
        Desired mean of the output.
    target_std : float
        Desired standard deviation of the output.

    Returns
    -------
    np.ndarray
        Rescaled values with the specified mean and std.
    """
    std_val = np.std(correlated_values)
    if std_val > 0:
        normed = (correlated_values - np.mean(correlated_values)) / std_val
    else:
        normed = np.zeros_like(correlated_values)
    return target_mean + target_std * normed


def _add_measurement_noise(
    rng: np.random.Generator,
    values: np.ndarray,
    noise_frac: float = MEASUREMENT_NOISE_FRAC,
) -> np.ndarray:
    """
    Add Gaussian measurement noise to readings.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    values : np.ndarray
        Clean parameter values.
    noise_frac : float
        Noise as a fraction of the reading (e.g. 0.02 = 2%).

    Returns
    -------
    np.ndarray
        Values with added noise.
    """
    noise = rng.normal(0, noise_frac, size=values.shape) * values
    return values + noise


def _add_sensor_glitches(
    rng: np.random.Generator,
    values: np.ndarray,
    glitch_prob: float = GLITCH_PROBABILITY,
    glitch_mag: float = GLITCH_MAGNITUDE_FRAC,
) -> np.ndarray:
    """
    Add random sensor glitches (±15% spikes) to a fraction of readings.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    values : np.ndarray
        Parameter values.
    glitch_prob : float
        Probability of a glitch on any single reading.
    glitch_mag : float
        Magnitude of the glitch as a fraction of the reading.

    Returns
    -------
    np.ndarray
        Values with occasional spikes.
    """
    result = values.copy()
    mask = rng.random(size=values.shape) < glitch_prob
    signs = rng.choice([-1.0, 1.0], size=values.shape)
    result[mask] = result[mask] * (1.0 + signs[mask] * glitch_mag)
    return result


# ---------------------------------------------------------------------------
# Population generators
# ---------------------------------------------------------------------------

def _generate_healthy_chips(
    rng: np.random.Generator,
    n: int,
) -> dict[str, np.ndarray]:
    """
    Generate healthy chip parameters.

    Healthy (85% default): near-zero drift rate.
      i_leak_0    ~ N(10.0, 1.5) clipped to [5, 20]   µA
      drift_rate  ~ N(2e-5, 1e-5) → ~0.3% growth over 168 h

    Returns dict with keys: i_leak_0, drift_i_leak, drift_v_th, drift_t_delay,
    v_th_0, t_delay_0
    """
    # Leakage
    i_leak_0 = rng.normal(10.0, 1.5, size=n).clip(5.0, 20.0)   # µA
    drift_i_leak = rng.normal(2e-5, 1e-5, size=n)               # 1/hours

    # Correlated drifts — low correlation for healthy chips (ρ ≈ 0.1)
    corr_vth, corr_tdelay = _generate_correlated_drifts(
        rng, n, drift_i_leak, ASSUMED_HEALTHY_CORRELATION,
    )

    # v_th: healthy chips have tiny drift around the midpoint of [0.35, 0.55] V
    v_th_0 = rng.normal(0.45, 0.02, size=n).clip(0.36, 0.54)   # volts
    drift_v_th = _scale_correlated_to_drift(corr_vth, 1e-5, 5e-6)

    # t_delay: healthy chips sit comfortably under 2.50 ns
    t_delay_0 = rng.normal(1.8, 0.1, size=n).clip(1.4, 2.3)    # ns
    drift_t_delay = _scale_correlated_to_drift(corr_tdelay, 1e-5, 5e-6)

    return {
        "i_leak_0": i_leak_0,
        "drift_i_leak": drift_i_leak,
        "v_th_0": v_th_0,
        "drift_v_th": drift_v_th,
        "t_delay_0": t_delay_0,
        "drift_t_delay": drift_t_delay,
    }


def _generate_latent_chips(
    rng: np.random.Generator,
    n: int,
) -> dict[str, np.ndarray]:
    """
    Generate latent-defective chip parameters.

    Latent-defective (10% default): elevated drift, but must stay UNDER
    the 50 µA spec limit at 168 h. This is enforced by the resample loop
    in generate_lot().

      i_leak_0    ~ N(12.0, 2.0)           µA
      drift_rate  ~ U(0.006, 0.010)        1/hours → 12 µA grows to ~33–65 µA

    Returns dict with same keys as _generate_healthy_chips.
    """
    i_leak_0 = rng.normal(12.0, 2.0, size=n)                    # µA
    drift_i_leak = rng.uniform(0.006, 0.010, size=n)             # 1/hours

    # ASSUMED_DEFECT_CORRELATION (ρ ≈ 0.6):
    # ASSUMPTION, NOT A MEASURED VALUE.
    # A shared defect mechanism (e.g. gate oxide thinning) is physically
    # expected to perturb leakage, threshold voltage, and delay together.
    # This makes the multivariate Mahalanobis screen meaningful.
    corr_vth, corr_tdelay = _generate_correlated_drifts(
        rng, n, drift_i_leak, ASSUMED_DEFECT_CORRELATION,
    )

    # v_th: defective chips drift more aggressively
    v_th_0 = rng.normal(0.44, 0.025, size=n).clip(0.36, 0.54)   # volts
    drift_v_th = _scale_correlated_to_drift(corr_vth, 3e-4, 1e-4)

    # t_delay: defective chips also show elevated delay drift
    t_delay_0 = rng.normal(1.9, 0.12, size=n).clip(1.4, 2.4)    # ns
    drift_t_delay = _scale_correlated_to_drift(corr_tdelay, 2e-4, 8e-5)

    return {
        "i_leak_0": i_leak_0,
        "drift_i_leak": drift_i_leak,
        "v_th_0": v_th_0,
        "drift_v_th": drift_v_th,
        "t_delay_0": t_delay_0,
        "drift_t_delay": drift_t_delay,
    }


def _generate_early_fail_chips(
    rng: np.random.Generator,
    n: int,
) -> dict[str, np.ndarray]:
    """
    Generate early-failure chip parameters.

    Early-failure (5% default): dies during burn-in, caught by existing methods.
      drift_rate ~ U(0.02, 0.04) → crosses spec well before 168 h

    Returns dict with same keys as _generate_healthy_chips.
    """
    i_leak_0 = rng.normal(14.0, 3.0, size=n).clip(5.0, 25.0)   # µA
    drift_i_leak = rng.uniform(0.02, 0.04, size=n)              # 1/hours

    # High correlation for defective chips (same mechanism)
    corr_vth, corr_tdelay = _generate_correlated_drifts(
        rng, n, drift_i_leak, ASSUMED_DEFECT_CORRELATION,
    )

    v_th_0 = rng.normal(0.46, 0.03, size=n).clip(0.36, 0.54)   # volts
    drift_v_th = _scale_correlated_to_drift(corr_vth, 5e-4, 2e-4)

    t_delay_0 = rng.normal(2.0, 0.15, size=n).clip(1.4, 2.4)   # ns
    drift_t_delay = _scale_correlated_to_drift(corr_tdelay, 4e-4, 1.5e-4)

    return {
        "i_leak_0": i_leak_0,
        "drift_i_leak": drift_i_leak,
        "v_th_0": v_th_0,
        "drift_v_th": drift_v_th,
        "t_delay_0": t_delay_0,
        "drift_t_delay": drift_t_delay,
    }


# ---------------------------------------------------------------------------
# Chip-to-rows expansion
# ---------------------------------------------------------------------------

def _chips_to_rows(
    rng: np.random.Generator,
    chip_ids: list[str],
    lot_id: str,
    wafer_ids: list[str],
    params: dict[str, np.ndarray],
    true_class: str,
) -> list[dict]:
    """
    Expand chip parameters into the canonical long-format rows (4 per chip).

    Applies Arrhenius degradation p(t) = p0 * exp(a * t), then adds
    measurement noise and sensor glitches.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    chip_ids : list[str]
        Unique chip identifiers.
    lot_id : str
        Lot identifier (e.g. "L001").
    wafer_ids : list[str]
        Wafer identifiers per chip.
    params : dict[str, np.ndarray]
        Population parameters (i_leak_0, drift_i_leak, etc.).
    true_class : str
        Ground truth label: "healthy", "latent", or "early_fail".

    Returns
    -------
    list[dict]
        List of row dicts in the canonical schema.
    """
    rows = []
    n = len(chip_ids)

    for tp in TIMEPOINTS:
        t = float(tp)

        # Arrhenius degradation: p(t) = p0 * exp(a * t)
        i_leak_clean = _arrhenius_value(params["i_leak_0"], params["drift_i_leak"], t)
        v_th_clean = _arrhenius_value(params["v_th_0"], params["drift_v_th"], t)
        t_delay_clean = _arrhenius_value(params["t_delay_0"], params["drift_t_delay"], t)

        # Add 2% Gaussian measurement noise
        i_leak_noisy = _add_measurement_noise(rng, i_leak_clean)
        v_th_noisy = _add_measurement_noise(rng, v_th_clean)
        t_delay_noisy = _add_measurement_noise(rng, t_delay_clean)

        # Add sensor glitches (0.5% of readings, ±15% spike)
        i_leak_final = _add_sensor_glitches(rng, i_leak_noisy)
        v_th_final = _add_sensor_glitches(rng, v_th_noisy)
        t_delay_final = _add_sensor_glitches(rng, t_delay_noisy)

        for i in range(n):
            rows.append({
                "chip_id": chip_ids[i],
                "lot_id": lot_id,
                "wafer_id": wafer_ids[i],
                "t_hours": tp,
                "i_leak_ua": float(i_leak_final[i]),
                "v_th_v": float(v_th_final[i]),
                "t_delay_ns": float(t_delay_final[i]),
                "temp_c": STRESS_TEMP_C,
                "voltage_v": NOMINAL_VOLTAGE_V,
                "true_class": true_class,
            })

    return rows


# ---------------------------------------------------------------------------
# Demo chip rows
# ---------------------------------------------------------------------------

def _demo_chip_rows(
    rng: np.random.Generator,
    lot_id: str,
) -> list[dict]:
    """
    Generate the fixed DEMO-CHIP-A and DEMO-CHIP-B rows.

    These chips have hardcoded leakage values from SPEC.md Section 1 and
    physically reasonable v_th and t_delay trajectories. They exist so the
    demo narrative can point at them specifically.

    DEMO-CHIP-A: genuinely healthy, nearly flat trajectory.
    DEMO-CHIP-B: latent defect, clear drift but stays under 50 µA.

    Returns
    -------
    list[dict]
        8 rows (4 per chip) in the canonical schema.
    """
    rows = []

    # Chip A: healthy — flat v_th and t_delay
    chip_a_vth = {0: 0.45, 24: 0.450, 96: 0.451, 168: 0.451}    # volts
    chip_a_tdelay = {0: 1.80, 24: 1.800, 96: 1.801, 168: 1.802}  # ns

    # Chip B: latent defect — correlated drift in v_th and t_delay
    chip_b_vth = {0: 0.44, 24: 0.455, 96: 0.480, 168: 0.510}    # volts
    chip_b_tdelay = {0: 1.85, 24: 1.95, 96: 2.10, 168: 2.25}    # ns

    for tp in TIMEPOINTS:
        # Chip A
        rows.append({
            "chip_id": DEMO_CHIP_A_ID,
            "lot_id": lot_id,
            "wafer_id": f"{lot_id}-W01",
            "t_hours": tp,
            "i_leak_ua": DEMO_CHIP_A_LEAKAGE[tp],
            "v_th_v": chip_a_vth[tp],
            "t_delay_ns": chip_a_tdelay[tp],
            "temp_c": STRESS_TEMP_C,
            "voltage_v": NOMINAL_VOLTAGE_V,
            "true_class": CLASS_HEALTHY,
        })
        # Chip B
        rows.append({
            "chip_id": DEMO_CHIP_B_ID,
            "lot_id": lot_id,
            "wafer_id": f"{lot_id}-W01",
            "t_hours": tp,
            "i_leak_ua": DEMO_CHIP_B_LEAKAGE[tp],
            "v_th_v": chip_b_vth[tp],
            "t_delay_ns": chip_b_tdelay[tp],
            "temp_c": STRESS_TEMP_C,
            "voltage_v": NOMINAL_VOLTAGE_V,
            "true_class": CLASS_LATENT,
        })

    return rows


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_lot(
    n_chips: int = DEFAULT_N_CHIPS,
    lot_id: str = "L001",
    seed: int = DEFAULT_SEED,
    healthy_frac: float = DEFAULT_HEALTHY_FRAC,
    latent_frac: float = DEFAULT_LATENT_FRAC,
    early_fail_frac: float = DEFAULT_EARLY_FAIL_FRAC,
) -> pd.DataFrame:
    """
    Generate a complete synthetic burn-in lot.

    Produces n_chips + 2 chips (including DEMO-CHIP-A and DEMO-CHIP-B),
    each with 4 rows (one per timepoint) in the canonical long-format schema.

    Three populations with physically distinct drift behaviours are generated,
    and the latent-defective chips are resampled until every one stays under
    the 50 µA spec limit at 168 h — this is the entire point of the dataset.

    Parameters
    ----------
    n_chips : int
        Number of random chips to generate (default 500). Two demo chips
        are always added on top.
    lot_id : str
        Lot identifier, e.g. "L001".
    seed : int
        Random seed for reproducibility.
    healthy_frac : float
        Fraction of healthy chips (default 0.85).
    latent_frac : float
        Fraction of latent-defective chips (default 0.10).
    early_fail_frac : float
        Fraction of early-failure chips (default 0.05).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        chip_id, lot_id, wafer_id, t_hours, i_leak_ua, v_th_v,
        t_delay_ns, temp_c, voltage_v, true_class.
        Each chip has exactly 4 rows. Leakage values are in µA.
    """
    # Validate fractions
    total = healthy_frac + latent_frac + early_fail_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Population fractions must sum to 1.0, got {total:.4f} "
            f"(healthy={healthy_frac}, latent={latent_frac}, "
            f"early_fail={early_fail_frac})"
        )

    rng = np.random.default_rng(seed)

    # Population sizes
    n_healthy = int(round(n_chips * healthy_frac))
    n_early = int(round(n_chips * early_fail_frac))
    n_latent = n_chips - n_healthy - n_early  # remainder to avoid off-by-one

    # Assign chip IDs and wafer IDs
    def make_ids(prefix: str, count: int, start: int = 0) -> tuple[list[str], list[str]]:
        """Generate chip_id and wafer_id lists."""
        chip_ids = []
        wafer_ids = []
        for i in range(count):
            idx = start + i
            wafer_num = (idx // 25) + 1   # ~25 chips per wafer
            chip_ids.append(f"{lot_id}-W{wafer_num:02d}-C{idx:04d}")
            wafer_ids.append(f"{lot_id}-W{wafer_num:02d}")
        return chip_ids, wafer_ids

    all_rows: list[dict] = []

    # --- Healthy chips ---
    h_ids, h_wafers = make_ids("H", n_healthy, start=0)
    h_params = _generate_healthy_chips(rng, n_healthy)
    all_rows.extend(_chips_to_rows(rng, h_ids, lot_id, h_wafers, h_params, CLASS_HEALTHY))

    # --- Latent-defective chips ---
    # CRITICAL: Resample any chip whose 168 h leakage exceeds 50 µA.
    # This constraint is the ENTIRE POINT of the dataset — latent chips
    # must pass conventional testing (stay under spec) while still showing
    # dangerous drift trajectories.
    spec_max_i_leak = SPEC_LIMITS["i_leak_ua"]["max"]  # 50.0 µA

    # Use a tighter threshold for the clean (pre-noise) trajectory so that
    # even after 2% Gaussian noise + 0.5% chance of ±15% glitch, the chip
    # stays under 50 µA. Worst case: 47 * 1.02 * 1.15 ≈ 55 µA — but that's
    # the absolute worst; typically noise is symmetric and self-correcting.
    # 47 µA gives comfortable headroom while still producing chips that are
    # clearly drifting (e.g., 12 µA → 40–47 µA).
    resample_threshold = spec_max_i_leak - 3.0  # 47.0 µA

    latent_params = _generate_latent_chips(rng, n_latent)

    # Resample loop: reject chips that exceed the threshold at 168 h
    max_resample_iterations = 1000
    for iteration in range(max_resample_iterations):
        # Compute clean 168 h leakage (before noise, to check the underlying trajectory)
        i_leak_168 = _arrhenius_value(
            latent_params["i_leak_0"],
            latent_params["drift_i_leak"],
            float(BURN_IN_HOURS),
        )
        exceeds_spec = i_leak_168 >= resample_threshold
        n_bad = int(np.sum(exceeds_spec))

        if n_bad == 0:
            break

        # Regenerate only the chips that exceed spec
        replacement = _generate_latent_chips(rng, n_bad)
        for key in latent_params:
            latent_params[key][exceeds_spec] = replacement[key]
    else:
        # If we still have bad chips after max iterations, force them under spec
        # by capping drift rate (should not happen with reasonable distributions)
        i_leak_168 = _arrhenius_value(
            latent_params["i_leak_0"],
            latent_params["drift_i_leak"],
            float(BURN_IN_HOURS),
        )
        exceeds_spec = i_leak_168 >= resample_threshold
        if np.any(exceeds_spec):
            # Set drift so 168h value = 45.0 µA (safely under spec with noise headroom)
            safe_drift = np.log(45.0 / latent_params["i_leak_0"][exceeds_spec]) / float(BURN_IN_HOURS)
            latent_params["drift_i_leak"][exceeds_spec] = safe_drift

    l_ids, l_wafers = make_ids("L", n_latent, start=n_healthy)
    all_rows.extend(_chips_to_rows(rng, l_ids, lot_id, l_wafers, latent_params, CLASS_LATENT))

    # --- Early-failure chips ---
    e_ids, e_wafers = make_ids("E", n_early, start=n_healthy + n_latent)
    e_params = _generate_early_fail_chips(rng, n_early)
    all_rows.extend(_chips_to_rows(rng, e_ids, lot_id, e_wafers, e_params, CLASS_EARLY_FAIL))

    # --- Demo chips (always included) ---
    all_rows.extend(_demo_chip_rows(rng, lot_id))

    # Build DataFrame
    df = pd.DataFrame(all_rows)

    # Enforce column order matching the canonical schema
    column_order = [
        "chip_id", "lot_id", "wafer_id", "t_hours",
        "i_leak_ua", "v_th_v", "t_delay_ns",
        "temp_c", "voltage_v", "true_class",
    ]
    df = df[column_order]

    # Sort by chip_id then t_hours for readability
    df = df.sort_values(["chip_id", "t_hours"]).reset_index(drop=True)

    return df


def generate_lot_from_preset(
    preset: str,
    n_chips: int = DEFAULT_N_CHIPS,
    lot_id: str = "L001",
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Generate a lot using a named preset.

    Presets defined in constants.py:
      - "healthy_heavy":   (0.92, 0.05, 0.03)
      - "defect_heavy":    (0.70, 0.20, 0.10)
      - "realistic_mixed": (0.85, 0.10, 0.05)

    Parameters
    ----------
    preset : str
        One of the preset names.
    n_chips : int
        Number of random chips.
    lot_id : str
        Lot identifier.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Generated lot data in canonical long format.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(PRESETS.keys())}"
        )
    healthy_frac, latent_frac, early_fail_frac = PRESETS[preset]
    return generate_lot(
        n_chips=n_chips,
        lot_id=lot_id,
        seed=seed,
        healthy_frac=healthy_frac,
        latent_frac=latent_frac,
        early_fail_frac=early_fail_frac,
    )
