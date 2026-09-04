"""
AGNIPARIKSHA — Streamlit Demo UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from src.generator import generate_lot
from src.pipeline import run_pipeline
from src.constants import (
    DEMO_CHIP_A_ID, DEMO_CHIP_B_ID, PRESETS, 
    SPEC_LIMITS, COL_I_LEAK, BURN_IN_HOURS,
    MISSION_LIFE_YEARS
)
from src.explain import explain_chip

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AGNIPARIKSHA", layout="wide")

st.title("AGNIPARIKSHA")
st.markdown("### Burn-in screening that reads the drift, not just the final reading")
st.markdown("**Team Bit2Byte, SIH26170**")

# ---------------------------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

data_source = st.sidebar.radio("Data Source", ["Generate synthetic lot", "Upload CSV/Parquet"])

if data_source == "Generate synthetic lot":
    n_chips = st.sidebar.slider("Chip count", min_value=100, max_value=2000, value=500, step=100)
    preset = st.sidebar.selectbox("Preset", list(PRESETS.keys()))
    seed = st.sidebar.number_input("Seed", value=42, step=1)
    uploaded_file = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload Data", type=["csv", "parquet", "pq"])
    n_chips = 0
    preset = "realistic_mixed"
    seed = 42

st.sidebar.markdown("---")
pat_sigma = st.sidebar.slider("PAT Sigma (n_sigma)", min_value=3.0, max_value=8.0, value=6.0, step=0.1)
alpha = st.sidebar.slider("Conformal Alpha", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
ea_val = st.sidebar.slider("Activation Energy (Ea) [eV]", min_value=0.6, max_value=1.1, value=0.7, step=0.05)
mission_life = st.sidebar.slider("Mission Life Requirement [years]", min_value=5, max_value=20, value=15, step=1)

run_button = st.sidebar.button("Run screening", type="primary")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def generate_rejection_pdf(chip_row: pd.Series) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    chip_id = chip_row["chip_id"]
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, f"REJECTION REPORT: {chip_id}")
    
    c.setFont("Helvetica", 12)
    y = 710
    c.drawString(50, y, f"Verdict: {chip_row['verdict']}")
    y -= 20
    c.drawString(50, y, f"Risk Score: {chip_row['risk_score']} / 4")
    y -= 20
    
    c.drawString(50, y, "Flags Triggered:")
    y -= 20
    for reason in chip_row["flag_reasons"]:
        c.drawString(70, y, f"- {reason}")
        y -= 20
        
    y -= 20
    c.drawString(50, y, f"168h Forecast: {chip_row['pred_168h']:.2f} µA")
    y -= 20
    lower = chip_row.get(f"pred_lower_{int((1-alpha)*100)}", 0)
    upper = chip_row.get(f"pred_upper_{int((1-alpha)*100)}", 0)
    c.drawString(50, y, f"95% Confidence Interval: [{lower:.2f}, {upper:.2f}] µA")
    
    y -= 20
    y07 = chip_row.get("life_years_ea07", 0)
    y10 = chip_row.get("life_years_ea10", 0)
    c.drawString(50, y, f"Projected Lifetime Range: {y07:.1f} to {y10:.1f} years")
    
    c.save()
    buffer.seek(0)
    return buffer.read()


def build_plotly_waterfall(shap_dict: dict):
    base_val = shap_dict["base_value"]
    shap_vals = shap_dict["shap_values"]
    feat_names = shap_dict["feature_names"]
    feat_vals = shap_dict["data"]
    
    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(shap_vals))
    # Keep top 8 features for clarity
    top_indices = indices[-8:]
    
    x_labels = []
    y_vals = []
    text_vals = []
    
    for idx in top_indices:
        x_labels.append(f"{feat_names[idx]}<br>({feat_vals[idx]:.2f})")
        y_vals.append(shap_vals[idx])
        text_vals.append(f"{shap_vals[idx]:+.2f}")
        
    # Add an "Other features" bucket
    other_sum = np.sum([shap_vals[i] for i in indices[:-8]])
    if abs(other_sum) > 0.01:
        x_labels.insert(0, "Other Features")
        y_vals.insert(0, other_sum)
        text_vals.insert(0, f"{other_sum:+.2f}")
        
    fig = go.Figure(go.Waterfall(
        name="SHAP", orientation="v",
        measure=["relative"] * len(y_vals) + ["total"],
        x=x_labels + ["Prediction"],
        textposition="outside",
        text=text_vals + [f"{base_val + sum(y_vals):.2f}"],
        y=y_vals + [base_val + sum(y_vals)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(
        title="SHAP Feature Attributions",
        showlegend=False,
        waterfallgap=0.3
    )
    return fig


# ---------------------------------------------------------------------------
# Explain Expander
# ---------------------------------------------------------------------------
with st.expander("How this works (The 8-Stage Pipeline)"):
    st.markdown("""
    **Conventional burn-in testing (PAT)** only checks if the final 168-hour leakage measurement is below the specification limit (e.g., 50 µA). This misses chips that drift rapidly but haven't crossed the line yet.
    
    **AGNIPARIKSHA** fixes this using an 8-stage pipeline:
    1. **Ingest**: Validates and formats the data.
    2. **PAT Screen**: Our baseline. Flags chips exceeding absolute limits.
    3. **Multivariate Screen**: Measures the *shape* of the drift trajectory across leakage, threshold voltage, and delay using Mahalanobis distance. Uncovers complex correlated defects.
    4. **Arrhenius Fit**: Fits early measurements (0h, 24h) to physical Arrhenius kinetics to anchor ML models in real silicon physics.
    5. **ML Forecast**: A Random Forest predicts the 168h value using early readings and the physics drift rates. Critically, it only trains on chips that pass PAT to focus strictly on subtle latent defects.
    6. **Lifetime Projection**: Extrapolates the failure time at operating conditions, presenting a conservative range that respects Activation Energy uncertainty.
    7. **Conformal Prediction**: Wraps the forecast in a statistically guaranteed 95% confidence bound.
    8. **Decision**: Synthesizes all flags. A chip is rejected if *any* rule fails, ensuring aerospace-grade conservatism.
    """)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if run_button:
    try:
        with st.spinner("Running AGNIPARIKSHA Pipeline..."):
            if data_source == "Generate synthetic lot":
                df_long = generate_lot(n_chips=n_chips, seed=seed)
                # Ensure the selected preset fractions are used (we have preset in selectbox, but generator takes kwargs)
                # Actually, generator defaults to realistic_mixed. We can just use the preset constants.
                hc, lc, efc = PRESETS[preset]
                df_long = generate_lot(n_chips=n_chips, seed=seed, healthy_frac=hc, latent_frac=lc, early_fail_frac=efc)
            else:
                if uploaded_file is None:
                    st.error("Please upload a file.")
                    st.stop()
                if uploaded_file.name.endswith(".csv"):
                    df_long = pd.read_csv(uploaded_file)
                else:
                    df_long = pd.read_parquet(uploaded_file)
            
            # Run the pipeline
            df, mapie_model, eval_report = run_pipeline(df_long, alpha=alpha, model_type="rf")
            
            st.session_state["df_res"] = df
            st.session_state["mapie_model"] = mapie_model
            st.session_state["eval_report"] = eval_report
            st.session_state["run_success"] = True
            
    except Exception as e:
        st.error(f"Pipeline Failed: {e}")
        st.session_state["run_success"] = False


# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
if st.session_state.get("run_success", False):
    df = st.session_state["df_res"]
    model = st.session_state["mapie_model"]
    report = st.session_state["eval_report"]
    
    st.markdown("---")
    
    # -----------------------------------------------------------------------
    # Section 1: Lot Summary
    # -----------------------------------------------------------------------
    st.header("1. Lot Summary")
    
    n_total = len(df)
    n_flagged = (df["verdict"] == "FLAG").sum()
    n_passed = n_total - n_flagged
    flag_rate = (n_flagged / n_total) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chips Screened", f"{n_total} chips")
    c2.metric("Passed", f"{n_passed} chips")
    c3.metric("Flagged", f"{n_flagged} chips")
    c4.metric("Flag Rate", f"{flag_rate:.1f} %")
    
    # Stacked bar of rules fired
    rule_counts = {
        "PAT Outlier": df["pat_flag"].sum(),
        "Mahalanobis Outlier": df["mv_flag"].sum(),
        "Conformal Exceedance": df["conformal_flag"].sum(),
        "Lifetime Shortfall": df["lifetime_flag"].sum(),
    }
    fig_rules = go.Figure(data=[
        go.Bar(name="Fired", x=list(rule_counts.keys()), y=list(rule_counts.values()))
    ])
    fig_rules.update_layout(title="Rules Triggered (A chip may trigger multiple)", yaxis_title="Number of Chips")
    st.plotly_chart(fig_rules, use_container_width=True)
    
    
    # -----------------------------------------------------------------------
    # Section 2: The Blind Spot (The Money Shot)
    # -----------------------------------------------------------------------
    st.header("2. The Blind Spot")
    st.markdown("*Conventional testing misses rapid drift that remains under the 50 µA absolute limit. Both highlighted chips pass PAT, but only one is safe.*")
    
    timepoints = [0, 24, 96, 168]
    leakage_cols = [f"i_leak_ua_{t}h" for t in timepoints]
    
    fig_traj = go.Figure()
    
    # Plot all chips in faint grey
    # To avoid rendering 2000 lines individually, we use a single scatter with Nones
    x_grey = []
    y_grey = []
    
    for idx in df.index:
        chip_id = df.loc[idx, "chip_id"]
        # Skip demo chips for the grey background
        if DEMO_CHIP_A_ID in chip_id or DEMO_CHIP_B_ID in chip_id:
            continue
            
        x_grey.extend(timepoints + [None])
        y_grey.extend(df.loc[idx, leakage_cols].values.tolist() + [None])
        
    fig_traj.add_trace(go.Scatter(
        x=x_grey, y=y_grey, 
        mode="lines", 
        line=dict(color="rgba(200, 200, 200, 0.1)"), 
        name="Other Chips",
        hoverinfo="skip"
    ))
    
    # Plot DEMO-CHIP-A (Green)
    demo_a = df[df["chip_id"].str.contains(DEMO_CHIP_A_ID)]
    if not demo_a.empty:
        a_row = demo_a.iloc[0]
        fig_traj.add_trace(go.Scatter(
            x=timepoints, y=a_row[leakage_cols],
            mode="lines+markers",
            line=dict(color="green", width=3),
            name="DEMO-CHIP-A (Healthy)"
        ))
        
    # Plot DEMO-CHIP-B (Amber)
    demo_b = df[df["chip_id"].str.contains(DEMO_CHIP_B_ID)]
    if not demo_b.empty:
        b_row = demo_b.iloc[0]
        fig_traj.add_trace(go.Scatter(
            x=timepoints, y=b_row[leakage_cols],
            mode="lines+markers",
            line=dict(color="orange", width=3),
            name="DEMO-CHIP-B (Latent Defect)"
        ))
        
    # Spec limit line
    spec_max = SPEC_LIMITS[COL_I_LEAK]["max"]
    fig_traj.add_hline(
        y=spec_max, line_dash="dash", line_color="red", 
        annotation_text=f"Spec Limit ({spec_max} µA)"
    )
    
    fig_traj.update_layout(
        xaxis_title="Burn-In Time [hours]",
        yaxis_title="Leakage Current [µA]",
        hovermode="closest"
    )
    st.plotly_chart(fig_traj, use_container_width=True)


    # -----------------------------------------------------------------------
    # Section 3: Screening Results
    # -----------------------------------------------------------------------
    st.header("3. Screening Results")
    
    filter_val = st.radio("Filter Verdict:", ["ALL", "FLAG", "PASS"], horizontal=True)
    
    df_disp = df.copy()
    if filter_val != "ALL":
        df_disp = df_disp[df_disp["verdict"] == filter_val]
        
    # Format the display dataframe
    df_disp["Projected Life"] = df_disp.apply(
        lambda r: f"{r['life_years_ea07']:.1f} - {r['life_years_ea10']:.1f} yrs", axis=1
    )
    conf_level = int((1-alpha)*100)
    df_disp["168h Forecast 95% CI"] = df_disp.apply(
        lambda r: f"[{r[f'pred_lower_{conf_level}']:.1f}, {r[f'pred_upper_{conf_level}']:.1f}] µA", axis=1
    )
    df_disp["Mahalanobis D²"] = df_disp["mahalanobis_d2"].round(1)
    
    cols_to_show = ["chip_id", "lot_id", "verdict", "risk_score", "flag_reasons", "168h Forecast 95% CI", "Projected Life", "Mahalanobis D²"]
    
    def color_verdict(val):
        color = 'red' if val == 'FLAG' else 'green'
        return f'color: {color}'
        
    st.dataframe(
        df_disp[cols_to_show].style.map(color_verdict, subset=['verdict']),
        use_container_width=True
    )

    
    # -----------------------------------------------------------------------
    # Section 4: Per-Chip Drill-Down
    # -----------------------------------------------------------------------
    st.header("4. Per-Chip Drill-Down")
    
    selected_chip = st.selectbox("Select a chip to inspect:", df["chip_id"].unique())
    chip_data = df[df["chip_id"] == selected_chip].iloc[0]
    
    st.subheader(f"Analysis for {selected_chip}")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Plot measured points, Arrhenius fit, and forecast point
        fig_drill = go.Figure()
        
        # Raw points
        fig_drill.add_trace(go.Scatter(
            x=timepoints, y=chip_data[leakage_cols],
            mode="markers", marker=dict(size=10, color="blue"),
            name="Measured [µA]"
        ))
        
        # Arrhenius curve (uses p0 and drift_rate_a)
        t_curve = np.linspace(0, 168, 100)
        p0 = chip_data[f"p0_fitted_{COL_I_LEAK}"]
        a = chip_data[f"drift_rate_a_{COL_I_LEAK}"]
        y_curve = p0 * np.exp(a * t_curve)
        
        fig_drill.add_trace(go.Scatter(
            x=t_curve, y=y_curve,
            mode="lines", line=dict(color="rgba(0,0,255,0.4)", dash="dot"),
            name="Arrhenius Fit"
        ))
        
        # ML Forecast with Error Bar
        pred = chip_data["pred_168h"]
        upper = chip_data[f"pred_upper_{conf_level}"]
        lower = chip_data[f"pred_lower_{conf_level}"]
        
        fig_drill.add_trace(go.Scatter(
            x=[168], y=[pred],
            mode="markers", marker=dict(size=12, color="purple", symbol="diamond"),
            name="ML Forecast [µA]",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[upper - pred],
                arrayminus=[pred - lower],
                color="purple"
            )
        ))
        
        fig_drill.add_hline(y=spec_max, line_dash="dash", line_color="red", annotation_text="Spec Limit")
        fig_drill.update_layout(title="Trajectory & Forecast", xaxis_title="Time [h]", yaxis_title="Leakage [µA]")
        st.plotly_chart(fig_drill, use_container_width=True)
        
    with col_b:
        # The Lifetime Block
        st.markdown("### Lifetime Assessment")
        
        st.info(f"**Projected life:** {chip_data['life_years_ea07']:.1f} – {chip_data['life_years_ea10']:.1f} years\n\n"
                f"*Range reflects activation-energy uncertainty, which we do not measure per chip.*")
                
        st.success(f"**Rank in lot:** {chip_data['life_rank_in_lot']:.2f}× median\n\n"
                   f"*Ea-independent, the reliable comparison metric.*")
                   
        st.warning(f"**168h Forecast 95% Interval:** [{lower:.1f}, {upper:.1f}] µA\n\n"
                   f"*Covers regression error only (ML uncertainty), NOT physical Ea uncertainty.*")
                   
        st.markdown(f"**Verdict:** {chip_data['verdict']}")
        if chip_data['verdict'] == "FLAG":
            st.error(f"**Triggered:** {', '.join(chip_data['flag_reasons'])}")

    st.markdown("### Model Explanation (SHAP)")
    # Generate SHAP explanation for the row
    from src.forecast import build_features
    # We must build the exact 14 features for this one row
    x_row = build_features(df[df["chip_id"] == selected_chip])
    explanation = explain_chip(model, x_row, list(x_row.columns))
    
    fig_shap = build_plotly_waterfall(explanation)
    st.plotly_chart(fig_shap, use_container_width=True)


    # -----------------------------------------------------------------------
    # Section 5: Performance vs Baseline
    # -----------------------------------------------------------------------
    if report is not None:
        st.header("5. Performance vs Baseline")
        
        rc1, rc2 = st.columns(2)
        
        with rc1:
            fig_recall = go.Figure(data=[
                go.Bar(name="Latent-Defect Recall", x=["Conventional PAT", "AGNIPARIKSHA"], 
                       y=[report.pat_recall_latent * 100, report.full_recall_latent * 100],
                       marker_color=["grey", "green"])
            ])
            fig_recall.update_layout(title="Recall on Latent Defects [%]", yaxis=dict(range=[0, 105]))
            st.plotly_chart(fig_recall, use_container_width=True)
            
        with rc2:
            st.markdown(f"**Conformal Coverage:** {report.conformal_coverage * 100:.1f} % *(Nominal: {conf_level} %)*")
            st.markdown(f"**F1 Score (Pipeline):** {report.full_f1:.3f}")
            st.markdown(f"**F1 Score (PAT):** {report.pat_f1:.3f}")
            
            st.markdown("#### Pipeline Confusion Matrix")
            st.dataframe(pd.DataFrame(report.full_conf_matrix, 
                                      columns=["Pred PASS", "Pred FLAG"], 
                                      index=["True PASS", "True DEFECT"]))


    # -----------------------------------------------------------------------
    # Section 6: Export
    # -----------------------------------------------------------------------
    st.header("6. Export Actions")
    
    ec1, ec2 = st.columns(2)
    
    with ec1:
        flagged_df = df[df["verdict"] == "FLAG"]
        csv = flagged_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Flagged Chips CSV",
            data=csv,
            file_name='flagged_chips.csv',
            mime='text/csv',
        )
        
    with ec2:
        if chip_data['verdict'] == "FLAG":
            pdf_bytes = generate_rejection_pdf(chip_data)
            st.download_button(
                label=f"Download Rejection Report ({selected_chip})",
                data=pdf_bytes,
                file_name=f'rejection_{selected_chip}.pdf',
                mime='application/pdf',
            )
        else:
            st.markdown(f"*(Rejection report not applicable for PASS chips)*")
