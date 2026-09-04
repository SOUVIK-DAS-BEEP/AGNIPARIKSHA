# BUILD PROMPT — AGNIPARIKSHA

> **This is the SPECIFICATION, not a task prompt.**
>
> Save this file into the repository as `SPEC.md`. The agent reads it as reference.
>
> Do NOT paste this whole document as a single build instruction — that produces one
> monolithic agent run with no checkpoints, and errors compound before you can catch
> them. Use `ANTIGRAVITY_PHASED_PROMPTS.md` instead: six sequential tasks, each ending
> in passing tests and a review checkpoint.
>
> Do not paraphrase or shorten what follows. Every constraint is deliberate.

---

## 0. WHAT YOU ARE BUILDING

Build **AGNIPARIKSHA** — a Python + Streamlit application that screens semiconductor
burn-in test data to identify chips that will fail after deployment, even though they
pass conventional testing today.

This is a **working prototype for a hackathon demo**. It must:

- run end-to-end on synthetic data with zero external setup
- be deployable to Streamlit Community Cloud so we can share a public demo URL
- be inspectable — a reliability engineer should be able to read the code and agree with it

It is **not** a research project. Do not invent new algorithms. Implement exactly the
pipeline specified in Section 4, using exactly the libraries in Section 2.

---

## 1. DOMAIN CONTEXT — READ THIS FIRST

You must understand the physical problem or the code will be subtly wrong.

**Burn-in screening.** Satellite electronics must be extremely reliable. Manufacturing
defects (thin oxide layers, contaminants, hairline cracks) can make a chip *latent-faulty*:
it passes initial tests but fails months later in orbit. To catch these, manufacturers
run **burn-in** — stressing chips at high temperature (125 °C) and voltage for one week
(168 hours) to accelerate ageing. Chips that die during burn-in are discarded.

**The gap we address.** Some chips survive burn-in and still meet spec limits, yet have
latent defects causing rapid parameter drift. Concretely, with a leakage-current spec
limit of 50 µA:

| Chip | 0 h | 24 h | 96 h | 168 h | Verdict today | Reality |
|------|-----|------|------|-------|---------------|---------|
| A    | 10.0 | 10.1 | 10.2 | 10.3 | PASS | genuinely healthy |
| B    | 12.0 | 22.0 | 35.0 | 45.0 | PASS | latent defect, will fail in orbit |

Both pass, because conventional testing only checks the **final reading** (45 < 50).
Chip B quadrupled in a week. Its **trajectory** is the danger signal, and nothing in
standard practice looks at trajectory.

**Why current industry practice misses it.** The standard is **Part Average Testing
(PAT)**, codified in AEC-Q001: for each parameter, compute a robust mean and robust
sigma across the lot, and reject anything outside ±6σ. Two weaknesses:

1. It is **static** — one snapshot, no trajectory.
2. It is **univariate** — each parameter screened separately, so it misses defects that
   show as *joint* drift across leakage AND threshold voltage AND timing together.

**Physics you will implement.** Degradation under constant thermal stress follows
Arrhenius kinetics. A parameter `p` drifts as:

```
p(t) = p₀ · exp(a · t)
```

where `a` is the chip's drift rate. The temperature dependence uses activation energy
`Ea` (0.7–1.0 eV for silicon). The acceleration factor between stress and use temperature:

```
AF = exp[ (Ea / k) · (1/T_use − 1/T_stress) ]

k        = 8.617e-5           eV/K   (Boltzmann constant)
T_stress = 398.15             K      (125 °C)
T_use    = 298.15             K      (25 °C, in-orbit operating)
Ea       = 0.7                eV     (default; make configurable)
```

With Ea = 0.7 this gives **AF ≈ 940**. So 168 h of burn-in ≈ 940 × 168 h ≈ 18 years of
room-temperature operation. Sanity check your implementation against AF ≈ 940 for
Ea = 0.7 — if you get a wildly different number, your unit conversion is wrong.

**CRITICAL — AF is extremely sensitive to Ea.** At Ea = 1.0 eV, still well inside the
normal silicon range, AF ≈ 17,600 — an **18× swing** in projected lifetime from a
parameter we do not measure per chip. Therefore:

- **Never report a single absolute lifetime as if it were precise.** Always report a
  range computed across Ea = 0.7 to 1.0 eV.
- **Also report a lot-relative rank**, i.e. this chip's projected life divided by its
  lot's median projected life. Ea cancels in that ratio, so the *ordering* of chips is
  robust even when absolute years are not. The ranking is the trustworthy output;
  the absolute range is context.
- The conformal interval in Stage 7 covers **regression error only**. It does NOT cover
  Ea uncertainty. Do not present a conformal interval around an absolute lifetime as
  though it accounted for both. Keep the two uncertainties visually and numerically
  separate in the UI.

---

## 2. TECH STACK — USE EXACTLY THESE

```
python           3.11
numpy            2.x
pandas           2.x
scipy            1.x
scikit-learn     1.4+
mapie            0.8+        # conformal prediction
shap             0.44+       # explainability
streamlit        1.31+
plotly           5.x         # all charts — do NOT use matplotlib in the UI
pyarrow                      # parquet I/O
pytest                       # tests
```

**Hard constraints:**

- **CPU only.** No GPU, no CUDA. Must run on a student laptop and on Streamlit
  Community Cloud's free tier.
- **No deep learning.** Do not import torch, tensorflow, or keras. If you think a neural
  network would help, you are misreading the spec — see Section 8.
- **No external APIs, no network calls at runtime.** Everything runs locally on
  uploaded or generated data.
- **No database.** Parquet files on disk are sufficient.
- Pin every version in `requirements.txt`.

---

## 3. DATA MODEL

### 3.1 Canonical schema

Every stage reads and writes this long-format schema. Do not change column names.

```
chip_id       str      unique per chip, e.g. "L003-W02-C0447"
lot_id        str      e.g. "L003"
wafer_id      str      e.g. "L003-W02"
t_hours       int      one of {0, 24, 96, 168}
i_leak_ua     float    leakage current, microamps
v_th_v        float    threshold voltage, volts
t_delay_ns    float    propagation delay, nanoseconds
temp_c        float    stress temperature, °C (125.0)
voltage_v     float    stress voltage, V (nominal 1.2)
```

So each chip contributes exactly **4 rows** (one per timepoint).

### 3.2 Spec limits (module-level constants, single source of truth)

```python
SPEC_LIMITS = {
    "i_leak_ua":  {"max": 50.0,  "min": None},
    "v_th_v":     {"max": 0.55,  "min": 0.35},
    "t_delay_ns": {"max": 2.50,  "min": None},
}
BURN_IN_HOURS      = 168
TIMEPOINTS         = [0, 24, 96, 168]
STRESS_TEMP_C      = 125.0
USE_TEMP_C         = 25.0
DEFAULT_EA_EV      = 0.7
BOLTZMANN_EV_PER_K = 8.617e-5
MISSION_LIFE_YEARS = 15.0   # ISRO GEO satellite design life; flag if projection is below
```

---

## 4. THE PIPELINE — EIGHT STAGES, IN THIS ORDER

Implement each stage as a separate module with a pure function. Each takes a DataFrame
and returns a DataFrame plus a result object. **Do not merge stages. Do not reorder.**

### Stage 1 — Ingest (`src/ingest.py`)

```python
def load_burn_in_data(path_or_buffer) -> pd.DataFrame
def validate_schema(df: pd.DataFrame) -> ValidationReport
def to_wide(df: pd.DataFrame) -> pd.DataFrame
```

- Accept CSV or Parquet.
- Validate: required columns present, exactly 4 timepoints per chip, no duplicate
  (chip_id, t_hours) pairs, numeric columns actually numeric.
- `to_wide` pivots to one row per chip with columns like `i_leak_ua_0h`,
  `i_leak_ua_24h`, `i_leak_ua_96h`, `i_leak_ua_168h`.
- On validation failure, return a report listing every problem. Never crash on bad input.

### Stage 2 — PAT screen (`src/pat.py`)

Univariate Part Average Testing per AEC-Q001. This is our **baseline**, not our
innovation — implement it faithfully so we can show what we improve on.

```python
def robust_stats(values: np.ndarray) -> tuple[float, float]
    # returns (median, robust_sigma)
    # robust_sigma = 1.4826 * MAD   ← the 1.4826 factor is required, it makes
    #                                  MAD a consistent estimator of sigma for
    #                                  normally distributed data. Do not omit it.

def pat_screen(df_wide, n_sigma=6.0) -> pd.DataFrame
```

- Group **by lot_id** — PAT limits are per-lot, never global.
- For each parameter, use the **final (168 h) reading** — that is what conventional PAT
  sees, and reproducing that limitation is the point.
- Limits: `median ± n_sigma * robust_sigma`.
- Output columns: `pat_flag` (bool), `pat_reason` (str, names which parameter),
  and per-parameter z-scores.

### Stage 3 — Multivariate outlier screen (`src/multivariate.py`)

**This is our first real contribution.** PAT is univariate and provably misses joint
drifts. We generalise it to multiple dimensions.

```python
def mahalanobis_screen(df_wide, contamination=0.02) -> pd.DataFrame
```

- Build a feature vector per chip from the **drift deltas**, not raw values:
  ```
  [ Δi_leak_0_24, Δi_leak_24_96, Δi_leak_96_168,
    Δv_th_0_24,   Δv_th_24_96,   Δv_th_96_168,
    Δt_delay_0_24, Δt_delay_24_96, Δt_delay_96_168 ]
  ```
  (9 features — deltas capture trajectory shape, which is exactly what PAT ignores.)
- Fit `sklearn.covariance.MinCovDet` **per lot** for a robust covariance estimate.
- Compute squared Mahalanobis distance per chip.
- Threshold at the chi-square critical value with `df = n_features` at α = 0.01,
  i.e. `scipy.stats.chi2.ppf(0.99, df=9)`. Use the chi-square threshold, not a
  percentile of the observed data — the theoretical threshold is defensible to a
  reviewer, an empirical one is not.
- Output: `mahalanobis_d2` (float), `mv_flag` (bool), and the top-contributing feature.
- Guard: if a lot has fewer chips than `2 * n_features`, MinCovDet is unstable — fall
  back to a diagonal covariance and set a `mv_low_confidence` flag. Do not let it crash.

### Stage 4 — Arrhenius fit (`src/arrhenius.py`)

```python
def fit_drift_rate(t_hours, values) -> tuple[float, float, float]
    # returns (p0, a, r_squared) for p(t) = p0 * exp(a*t)
```

- Fit using **only the 0 h and 24 h readings**. This is deliberate: the whole value
  proposition is predicting the outcome from *early* measurements, so the model must
  never see 96 h or 168 h at fit time.
- Implement as a log-linear least-squares fit: `ln(p) = ln(p0) + a·t`. Do not use
  `curve_fit` for this — the log-linear form is closed-form, faster and more stable.
- Handle non-positive values (log undefined): clamp to a small epsilon and set a
  `fit_degraded` flag rather than returning NaN.
- Output per chip: `drift_rate_a`, `p0_fitted`, `fit_r2` for each parameter.

### Stage 5 — ML drift forecast (`src/forecast.py`)

```python
def build_features(df_wide) -> pd.DataFrame
def train_forecaster(X_train, y_train) -> sklearn estimator
def predict_168h(model, X) -> np.ndarray
```

**Target:** the true 168 h leakage value.

**Features (never include any 96 h or 168 h measurement — that would leak the label):**

```
i_leak_ua_0h, i_leak_ua_24h, delta_i_leak_0_24,
v_th_v_0h, v_th_v_24h, delta_v_th_0_24,
t_delay_ns_0h, t_delay_ns_24h, delta_t_delay_0_24,
drift_rate_a_ileak,        # from Stage 4 — the physics anchor
lot_median_i_leak_0h, lot_std_i_leak_0h,
temp_c, voltage_v
```

- Models: `RandomForestRegressor(n_estimators=300, random_state=42)` and
  `SVR(kernel="rbf")`. Train both, report both, use RF as the default (SHAP is fast and
  exact on tree models).
- Split **by lot**, using `GroupKFold(n_splits=5)` grouped on `lot_id`. A random row
  split would leak lot-level information and inflate the score.
- Report MAE, RMSE, R² on held-out lots.

### Stage 6 — Lifetime projection (`src/lifetime.py`)

**Our second contribution.** Converts a burn-in forecast into a mission-relevant number.

```python
def acceleration_factor(ea_ev=0.7, t_stress_c=125.0, t_use_c=25.0) -> float

def project_lifetime_years(p0, drift_rate_a, spec_limit, af) -> float
    # single-Ea projection — internal helper only, never surfaced directly

def project_lifetime_range(p0, drift_rate_a, spec_limit,
                           ea_low=0.7, ea_high=1.0) -> tuple[float, float]
    # returns (years_at_ea_low, years_at_ea_high) — THIS is what the UI shows

def lifetime_rank_in_lot(df_lot) -> pd.Series
    # projected_life / lot median projected_life
    # Ea cancels in this ratio, so the rank is Ea-independent.
    # This is the primary, trustworthy lifetime output.
```

Derivation to implement:

```
At stress temperature, the chip crosses spec at:
    t_fail_stress = ln(spec_limit / p0) / a          [hours]

Convert to use temperature:
    t_fail_use = t_fail_stress * AF                  [hours]

    lifetime_years = t_fail_use / (24 * 365.25)
```

- If `a <= 0` (chip is stable or improving), lifetime is effectively unbounded — cap the
  reported value at 50 years and set `lifetime_capped = True`. Do not return infinity.
- Verify: `acceleration_factor(0.7)` must return ≈ 940 AND `acceleration_factor(1.0)`
  must return ≈ 17,600. Assert BOTH in a unit test — the second assertion is what proves
  you have understood the sensitivity.
- Output columns: `life_years_ea07`, `life_years_ea10`, `life_rank_in_lot`,
  `lifetime_flag`.
- **`lifetime_flag` fires on the CONSERVATIVE end of the range** — i.e. use
  `life_years_ea07` (the shorter projection) against `MISSION_LIFE_YEARS`. In aerospace,
  assume the worse case.
- There is deliberately **no** `projected_life_years` scalar column. If you find yourself
  wanting one, re-read the Ea warning in Section 1.

### Stage 7 — Conformal prediction (`src/conformal.py`)

**Our third contribution.** Gives a distribution-free coverage guarantee, which is what
makes the flag defensible rather than heuristic.

```python
def fit_conformal(base_model, X_cal, y_cal) -> MapieRegressor
def predict_with_interval(mapie, X, alpha=0.05) -> tuple[np.ndarray, np.ndarray]
```

- Use `mapie.regression.MapieRegressor` with `method="plus"` and `cv=5`.
- `alpha=0.05` gives 95% coverage. Expose alpha in the UI.
- **The decision uses the upper bound, not the point prediction.** A chip is flagged if
  the 95% upper bound on its predicted 168 h leakage exceeds the spec limit. This is
  deliberately conservative: in aerospace a false negative can end a mission, a false
  positive costs one chip.
- Output: `pred_168h`, `pred_lower_95`, `pred_upper_95`, `conformal_flag`.
- Validate empirical coverage on the held-out set and display it — if the guarantee says
  95% and we observe 94–96%, that is evidence the method works.

### Stage 8 — Decision and SHAP (`src/decide.py`, `src/explain.py`)

```python
def make_decision(df) -> pd.DataFrame
def explain_chip(model, X_row, feature_names) -> dict
```

**Decision rule — a chip is FLAGGED if ANY of these is true:**

```
1. pat_flag           — univariate outlier vs its lot
2. mv_flag            — multivariate trajectory outlier
3. conformal_flag     — 95% upper bound on 168 h forecast exceeds spec
4. lifetime_flag      — projected in-orbit life < MISSION_LIFE_YEARS
```

Record **which** rules fired in `flag_reasons` (a list). Chips passing all four are
`PASS`. Assign `risk_score` = number of rules fired (0–4) for sorting.

For SHAP, use `shap.TreeExplainer` on the RandomForest. Generate a per-chip waterfall
for any flagged chip. Cache the explainer — do not rebuild it per chip.

---

## 5. SYNTHETIC DATA GENERATOR (`src/generator.py`)

Real ISRO burn-in data is unavailable, so we generate physically faithful data. This is
also an open-source contribution in its own right, so make it clean.

```python
def generate_lot(n_chips=500, lot_id="L001", seed=42,
                 healthy_frac=0.85, latent_frac=0.10, early_fail_frac=0.05
                 ) -> pd.DataFrame
```

Three chip populations:

**Healthy (85%)** — near-zero drift rate.
```
i_leak_0    ~ N(10.0, 1.5)   clipped to [5, 20]
drift_rate  ~ N(2e-5, 1e-5)  → ~0.3% growth over 168 h
```

**Latent-defective (10%)** — the chips we exist to catch. They must **stay under spec at
168 h** but have clearly elevated drift.
```
i_leak_0    ~ N(12.0, 2.0)
drift_rate  ~ U(0.006, 0.010)   → 12 µA grows to ~33–65 µA
```
After generating, **reject and resample any latent chip whose 168 h value exceeds
50 µA** — if it exceeds spec it would be caught by conventional testing and is not the
population we care about. This constraint is the entire point of the dataset. Do not
skip it.

**Early-failure (5%)** — dies during burn-in, caught by existing methods.
```
drift_rate ~ U(0.02, 0.04)     → crosses spec before 168 h
```

Additional requirements:

- Correlate `v_th_v` and `t_delay_ns` drift with leakage drift at ρ ≈ 0.6 for defective
  chips, ρ ≈ 0.1 for healthy ones. **This is what makes the multivariate stage
  meaningful** — without correlation, Mahalanobis adds nothing over PAT.
- **Label this honestly in the code.** ρ = 0.6 is a physically plausible ASSUMPTION
  (a shared defect mechanism should perturb several parameters together), not a
  measured value from real silicon. Put exactly that in a comment above the constant,
  name it `ASSUMED_DEFECT_CORRELATION`, and expose it as a tunable parameter so a
  reviewer can vary it. Report in the README how detection recall changes as ρ falls
  to 0.3 and 0.1 — if the method only works at high ρ, we need to know that and say so.
- Add measurement noise: 2% Gaussian on every reading.
- Add sensor glitches: 0.5% of readings get a ±15% spike.
- Emit a hidden ground-truth column `true_class` ∈ {healthy, latent, early_fail} for
  evaluation. The pipeline must **never** read this column — only the evaluation module may.
- Ship three presets: `healthy_heavy`, `defect_heavy`, `realistic_mixed`.

**Include Chip A and Chip B from Section 1 as fixed, named chips in every generated lot**
(`chip_id = "DEMO-CHIP-A"` and `"DEMO-CHIP-B"`) with exactly the values in that table.
The demo narrative depends on being able to point at them.

---

## 6. EVALUATION (`src/evaluate.py`)

Because we have ground truth on synthetic data, prove the pipeline works.

```python
def evaluate(df_results, df_truth) -> EvalReport
```

Report:

- **Recall on latent-defective chips** — the headline metric. Missing one is a mission risk.
- Precision, F1, ROC-AUC.
- **Baseline comparison:** PAT alone vs PAT + our full pipeline. Show the improvement in
  latent-defect recall. This is the single most important number in the demo.
- Confusion matrix.
- Empirical conformal coverage vs nominal 95%.
- Full precision-recall curve, so ISRO can choose its own operating point.

Optimise for **recall**, not accuracy. State this explicitly in the UI.

---

## 7. STREAMLIT APP (`app/streamlit_app.py`)

Single page, top to bottom. Use `st.session_state` to hold results between interactions.

**Header** — "AGNIPARIKSHA" with subtitle "Burn-in screening that reads the drift, not
just the final reading". Team Bit2Byte, SIH26170.

**Sidebar**
- Data source: `Generate synthetic lot` (default) or `Upload CSV/Parquet`
- If generating: chip count slider (100–2000), preset dropdown, seed input
- PAT sigma slider (3.0–8.0, default 6.0)
- Conformal alpha slider (0.01–0.20, default 0.05)
- Activation energy Ea slider (0.6–1.1 eV, default 0.7)
- Mission life requirement (5–20 years, default 15)
- `Run screening` button

**Section 1 — Lot summary.** Four metric cards: chips screened, passed, flagged, flag
rate. Below them, a stacked bar of which rules fired.

**Section 2 — The blind spot.** Plotly line chart of all chips' leakage trajectories,
faint grey, with DEMO-CHIP-A in green and DEMO-CHIP-B in amber, plus a red dashed line
at the 50 µA spec. Caption: "Both chips pass conventional testing. Only one is safe."
**This is the money shot of the demo — make it look good.**

**Section 3 — Screening results.** Sortable dataframe: chip_id, lot, verdict, risk_score,
flag_reasons, pred_168h with 95% interval, projected_life_years, Mahalanobis d². Colour
flagged rows. Filter control for verdict.

**Section 4 — Per-chip drill-down.** Select a chip → show its four measured points, the
fitted Arrhenius curve, the forecast with conformal band, its SHAP waterfall, and the
lifetime block.

The lifetime block must show **three things, clearly separated**:
1. `Projected life: 4 – 71 years` (the Ea = 0.7 to 1.0 range) with the caption
   "range reflects activation-energy uncertainty, which we do not measure per chip"
2. `Rank in lot: 0.18× median` — labelled "Ea-independent, the reliable comparison"
3. The conformal interval on the **168 h forecast** — labelled explicitly as covering
   regression error only, NOT the Ea uncertainty above

Never render a single lifetime number with a ± attached to it. That would imply a
precision we do not have.

**Section 5 — Performance vs baseline.** Side-by-side bar chart: PAT-only recall vs
full-pipeline recall on latent-defective chips. Confusion matrices. Conformal coverage
check.

**Section 6 — Export.** Download flagged-chip CSV; download a per-chip PDF rejection
report (use `reportlab`).

**UI rules:**
- Every number displayed must carry a unit.
- Show a spinner during the run; never block silently.
- If a stage fails, show which stage and why, and keep earlier results visible.
- Include a "How this works" expander explaining the eight stages in plain English —
  judges from non-ML backgrounds will read it.

---

## 8. EXPLICITLY OUT OF SCOPE

Do **not** build these. They were considered and deliberately rejected. If you add them
you will break the demo narrative and contradict our submitted deck.

| Not building | Why |
|---|---|
| PINN (physics-informed neural net) | Documented as a stretch goal, not core scope. |
| Symbolic regression / PySR | Circular on synthetic data — we would "discover" the Arrhenius law we used to generate it. Only meaningful on real fab data. |
| LSTM autoencoder | Four timepoints is far too short a sequence to justify an LSTM. Mahalanobis on drift deltas does this job honestly. |
| Weibull survival / RUL fitting | Chips that pass burn-in produce no failure events — heavily censored data makes the fit unreliable. Deterministic Arrhenius projection replaces it. |
| User accounts, auth, database | Demo scope. |
| Cloud storage, external APIs | Must run offline. |

---

## 9. PROJECT STRUCTURE

```
agnipariksha/
├── README.md
├── requirements.txt
├── Makefile                  # `make demo` runs the full pipeline end-to-end
├── src/
│   ├── __init__.py
│   ├── constants.py          # SPEC_LIMITS, TIMEPOINTS, temperatures, Ea
│   ├── generator.py          # Stage 0 — synthetic data
│   ├── ingest.py             # Stage 1
│   ├── pat.py                # Stage 2
│   ├── multivariate.py       # Stage 3
│   ├── arrhenius.py          # Stage 4
│   ├── forecast.py           # Stage 5
│   ├── lifetime.py           # Stage 6
│   ├── conformal.py          # Stage 7
│   ├── decide.py             # Stage 8
│   ├── explain.py            # SHAP
│   ├── evaluate.py           # metrics vs ground truth
│   └── pipeline.py           # orchestrates 1→8, single entry point
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_arrhenius.py     # MUST assert AF(0.7)≈940 AND AF(1.0)≈17600
│   ├── test_pat.py           # MUST assert robust_sigma uses the 1.4826 factor
│   ├── test_generator.py     # MUST assert no latent chip exceeds spec at 168 h
│   ├── test_pipeline.py      # end-to-end smoke test
│   └── test_no_leakage.py    # MUST assert no 96h/168h column appears in features
├── data/synthetic/           # generated parquet, gitignored
└── notebooks/                # EDA scratch, not part of the app
```

---

## 10. BUILD ORDER

Build and verify in this order. Do not start a stage until the previous one has a
passing test. The six build tasks in `ANTIGRAVITY_PHASED_PROMPTS.md` group these
steps into phases with a human review checkpoint between each.

1. `constants.py` and `generator.py` → produce a lot, plot it, eyeball that Chip B
   looks like the table in Section 1
2. `ingest.py` → schema validation with deliberately broken inputs
3. `pat.py` → verify it catches early-failure chips and misses latent ones
   *(that miss is the expected result and the whole premise — assert it)*
4. `arrhenius.py` → verify AF ≈ 940 for Ea = 0.7
5. `multivariate.py` → verify it catches latent chips PAT missed
6. `forecast.py` → verify no label leakage, report GroupKFold metrics
7. `lifetime.py` → verify Chip B projects to a short life, Chip A to a long one
8. `conformal.py` → verify empirical coverage lands near 95%
9. `decide.py` + `explain.py` → end-to-end verdicts
10. `evaluate.py` → the PAT-vs-us comparison
11. `streamlit_app.py` → wire it all together

---

## 11. DEFINITION OF DONE

- [ ] `make demo` runs the full pipeline on a generated lot with no errors
- [ ] `pytest` passes, including all five named assertions in Section 9
- [ ] No UI element shows a single absolute lifetime with a ± interval attached
- [ ] Streamlit app runs locally with `streamlit run app/streamlit_app.py`
- [ ] The app catches DEMO-CHIP-B and passes DEMO-CHIP-A
- [ ] Latent-defect recall is measurably higher for the full pipeline than for PAT alone
- [ ] Empirical conformal coverage is within ±3% of nominal
- [ ] No feature used by the forecaster contains any 96 h or 168 h measurement
- [ ] Deploys to Streamlit Community Cloud from a public GitHub repo
- [ ] README documents the eight stages and the physics, with the AF derivation shown

---

## 12. CODE STANDARDS

- Type hints on every public function.
- Docstrings stating units — `"""Returns leakage in microamps."""` — unit confusion is
  the most likely source of a silent physics bug here.
- `random_state=42` everywhere reproducibility matters.
- No magic numbers in logic; everything lives in `constants.py`.
- Every physics formula gets a comment naming what it is and where it comes from.
- Fail loudly with informative messages. Never silently return NaN.

---

## FINAL INSTRUCTION

If any part of this specification seems to conflict with itself, **stop and ask** rather
than guessing. Do not substitute a "better" algorithm for a specified one. The pipeline
described here was chosen against specific published precedents (AEC-Q001 for PAT,
IEEE IRPS for multivariate latent-defect screening, MAPIE for conformal prediction), and
deviating from it breaks the alignment between this prototype and our submitted proposal.
