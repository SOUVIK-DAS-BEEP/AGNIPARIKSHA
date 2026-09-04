# AGNIPARIKSHA

**Burn-in screening that reads the drift, not just the final reading.**

*Built by Team Bit2Byte (SIH26170)*

## The Problem (The Blind Spot)
Conventional aerospace burn-in testing relies heavily on PAT (Part Average Testing). PAT checks if a chip's final measurement at 168 hours is below a static specification limit (e.g., 50 µA).

This creates a dangerous blind spot. A chip can degrade rapidly from 10 µA to 45 µA during testing. Because 45 µA is less than 50 µA, PAT passes the chip. However, this chip possesses a latent defect and will almost certainly fail early in its mission life.

## The AGNIPARIKSHA Solution
AGNIPARIKSHA fixes this blind spot using an 8-stage pipeline combining multivariate statistics, machine learning, and physical silicon kinetics:

1. **Ingest**: Schema validation and data formatting.
2. **PAT Screen**: The standard univariate absolute-limit baseline.
3. **Multivariate Screen**: Mahalanobis distance profiling of the *shape* of the drift trajectory across leakage, threshold voltage, and delay, detecting subtle correlated defects.
4. **Arrhenius Fit**: Early measurements (0h, 24h) are fitted to Arrhenius kinetic physics to anchor forecasts.
5. **ML Forecast**: A Random Forest predicts the 168h end-state using early readings. Crucially, it trains *only* on PAT survivors, isolating the regression on the hardest subtle defects.
6. **Lifetime Projection**: Extrapolates failure times at mission conditions. Renders a conservative range (reflecting Ea uncertainty) and an Ea-independent lot rank.
7. **Conformal Prediction**: Wraps the forecast in a mathematically guaranteed 95% confidence upper bound to prevent false negatives.
8. **Decision Engine**: Rejects any chip that triggers *any* of the screening flags, ensuring maximum aerospace conservatism.

## Repository Structure

- `src/` — The core 8-stage pipeline modules.
- `tests/` — Comprehensive test suite (pytest), including the critical adversarial leakage tests.
- `app/` — The Streamlit application.
- `demo.py` & `Makefile` — CLI utilities for reproducing the 1000-chip benchmark.

## Running the Application

Ensure you have Python 3.11 installed.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. **CLI Benchmark:**
   ```bash
   make demo
   ```

## Assumptions and Limitations

We built this pipeline prioritizing data honesty and physical realism. As such, the system has the following known assumptions and limitations:

1. **Activation Energy (Ea) Uncertainty:** We assume true activation energy lies between 0.7 eV and 1.0 eV. Because we cannot measure Ea per chip without destructive testing, we do not project single scalar lifetimes. If a chip's true Ea falls outside this range, the conservative lower bound on the lifetime may be inaccurate.
2. **Accelerated Aging Models:** The Arrhenius equation assumes a single dominant failure mechanism. If multiple competing defect mechanisms exist with wildly different temperature dependencies, the log-linear extrapolation breaks down.
3. **Small Lot Instability:** The Mahalanobis multivariate screen relies on `MinCovDet`. For extremely small lots ($N < 50$), this estimator becomes unstable, forcing the pipeline to fall back to a diagonal covariance matrix, which drastically reduces its ability to catch correlated cross-parameter defects.
4. **Endpoint Limitation:** The model deliberately predicts the 168h endpoint to match legacy PAT capabilities. It assumes that capturing the true 168h drift accurately is sufficient to extrapolate the entire lifetime.
5. **Static Mission Profile:** The lifetime projection assumes a constant 50°C use condition. It does not natively handle cyclic mission thermal profiles.
