"""
AGNIPARIKSHA — Single source of truth for all project constants.

Every numeric constant used anywhere in the project is defined here.
No magic numbers anywhere else — import from this module.

All values come from SPEC.md Section 3.2 and Section 1.
"""

# ---------------------------------------------------------------------------
# Spec limits — the pass/fail thresholds for each measured parameter.
# These are industry-standard limits for semiconductor qualification.
# ---------------------------------------------------------------------------
SPEC_LIMITS: dict[str, dict[str, float | None]] = {
    "i_leak_ua":  {"max": 50.0,  "min": None},   # leakage current, microamps
    "v_th_v":     {"max": 0.55,  "min": 0.35},    # threshold voltage, volts
    "t_delay_ns": {"max": 2.50,  "min": None},    # propagation delay, nanoseconds
}

# ---------------------------------------------------------------------------
# Burn-in test parameters
# ---------------------------------------------------------------------------
BURN_IN_HOURS: int = 168                  # 1 week of stress testing
TIMEPOINTS: list[int] = [0, 24, 96, 168]  # measurement times in hours

# ---------------------------------------------------------------------------
# Temperature constants
# ---------------------------------------------------------------------------
STRESS_TEMP_C: float = 125.0   # burn-in stress temperature, °C
USE_TEMP_C: float = 25.0       # nominal in-orbit operating temperature, °C

# Conversions to Kelvin (for Arrhenius calculations)
STRESS_TEMP_K: float = STRESS_TEMP_C + 273.15   # 398.15 K
USE_TEMP_K: float = USE_TEMP_C + 273.15         # 298.15 K

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
DEFAULT_EA_EV: float = 0.7          # default activation energy, electronvolts
BOLTZMANN_EV_PER_K: float = 8.617e-5  # Boltzmann constant, eV/K

# ---------------------------------------------------------------------------
# Mission parameters
# ---------------------------------------------------------------------------
MISSION_LIFE_YEARS: float = 15.0    # ISRO GEO satellite design life, years

# ---------------------------------------------------------------------------
# Stress voltage (nominal, included in the data schema)
# ---------------------------------------------------------------------------
NOMINAL_VOLTAGE_V: float = 1.2      # nominal stress voltage, volts

# ---------------------------------------------------------------------------
# Canonical column names — avoids typos in string literals across modules
# ---------------------------------------------------------------------------
COL_CHIP_ID = "chip_id"
COL_LOT_ID = "lot_id"
COL_WAFER_ID = "wafer_id"
COL_T_HOURS = "t_hours"
COL_I_LEAK = "i_leak_ua"
COL_V_TH = "v_th_v"
COL_T_DELAY = "t_delay_ns"
COL_TEMP = "temp_c"
COL_VOLTAGE = "voltage_v"
COL_TRUE_CLASS = "true_class"

# ---------------------------------------------------------------------------
# DEMO chip values — hardcoded from SPEC.md Section 1, Table.
# Used to insert recognisable reference chips into every generated lot.
# ---------------------------------------------------------------------------
DEMO_CHIP_A_ID = "DEMO-CHIP-A"
DEMO_CHIP_B_ID = "DEMO-CHIP-B"

# Chip A: genuinely healthy, nearly flat leakage trajectory
DEMO_CHIP_A_LEAKAGE = {0: 10.0, 24: 10.1, 96: 10.2, 168: 10.3}   # µA

# Chip B: latent defect, rapidly drifting but still under 50 µA at 168 h
DEMO_CHIP_B_LEAKAGE = {0: 12.0, 24: 22.0, 96: 35.0, 168: 45.0}   # µA

# ---------------------------------------------------------------------------
# Generator population labels
# ---------------------------------------------------------------------------
CLASS_HEALTHY = "healthy"
CLASS_LATENT = "latent"
CLASS_EARLY_FAIL = "early_fail"

# ---------------------------------------------------------------------------
# Generator defaults
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_N_CHIPS = 500
DEFAULT_HEALTHY_FRAC = 0.85
DEFAULT_LATENT_FRAC = 0.10
DEFAULT_EARLY_FAIL_FRAC = 0.05

# Measurement noise: 2% Gaussian on every reading (SPEC.md Section 5)
MEASUREMENT_NOISE_FRAC = 0.02

# Sensor glitches: 0.5% of readings get a ±15% spike (SPEC.md Section 5)
GLITCH_PROBABILITY = 0.005
GLITCH_MAGNITUDE_FRAC = 0.15

# ---------------------------------------------------------------------------
# ASSUMED_DEFECT_CORRELATION
#
# ASSUMPTION, NOT A MEASURED VALUE.
#
# A shared defect mechanism (e.g., gate oxide thinning, contaminant diffusion)
# is physically expected to perturb several parameters together — leakage,
# threshold voltage, and propagation delay should co-drift. We assume ρ ≈ 0.6
# for defective chips and ρ ≈ 0.1 for healthy ones. This correlation is what
# makes the multivariate Mahalanobis screen (Stage 3) meaningful — without it,
# Mahalanobis adds nothing over univariate PAT.
#
# This value is physically plausible but not derived from real silicon
# measurements. It should be treated as a tunable parameter. See README for
# a sensitivity analysis showing how detection recall changes as ρ falls
# to 0.3 and 0.1.
# ---------------------------------------------------------------------------
ASSUMED_DEFECT_CORRELATION: float = 0.6
ASSUMED_HEALTHY_CORRELATION: float = 0.1

# ---------------------------------------------------------------------------
# Lifetime projection constants
# ---------------------------------------------------------------------------
LIFETIME_CAP_YEARS: float = 50.0   # cap for stable/improving chips
HOURS_PER_YEAR: float = 24.0 * 365.25  # hours in a year

# ---------------------------------------------------------------------------
# Generator presets — (healthy_frac, latent_frac, early_fail_frac)
# ---------------------------------------------------------------------------
PRESETS: dict[str, tuple[float, float, float]] = {
    "healthy_heavy":    (0.92, 0.05, 0.03),
    "defect_heavy":     (0.70, 0.20, 0.10),
    "realistic_mixed":  (0.85, 0.10, 0.05),
}
