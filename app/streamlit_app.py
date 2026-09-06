"""
AGNIPARIKSHA — Streamlit Demo UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import io
import sys
from pathlib import Path

# Ensure project root is on sys.path when running from app/ subdirectory
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
st.set_page_config(page_title="AGNIPARIKSHA", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for the UI Redesign
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, .stMetric label {
        font-family: 'Rajdhani', sans-serif !important;
    }
    
    /* Stepper */
    .stepper {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        color: #94a3b8;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stepper .active {
        color: #f59e0b;
        font-weight: bold;
    }
    
    /* Custom Metric Cards */
    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricValue"] {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: bold;
    }
    
    /* Glow Run Button */
    [data-testid="stBaseButton-primary"] {
        background-color: #f59e0b !important;
        color: #0a0f1e !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.4) !important;
        transition: all 0.3s ease;
    }
    [data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.8) !important;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("🔥 AGNIPARIKSHA")
    st.markdown("### Burn-in Screening Intelligence")
with col_h2:
    if st.button("⚙ Reset Layout", key="reset"):
        st.session_state["run_success"] = False
        st.rerun()


# Pipeline Stepper State
run_success = st.session_state.get("run_success", False)
if run_success:
    st.markdown("""
    <div class="stepper">
        <span>Configure</span> ──── <span>Run</span> ──○── <span class="active">Analyze</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="stepper">
        <span class="active">Configure</span> ──○── <span>Run</span> ──── <span>Analyze</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
if not run_success:
    st.markdown("---")
    col_config1, col_config2 = st.columns(2)

    with col_config1:
        st.subheader("Data Source")
        data_source = st.radio("Data Source", ["Generate synthetic lot", "Upload CSV/Parquet"], horizontal=True, label_visibility="collapsed")
        
        st.markdown("---")
        if data_source == "Generate synthetic lot":
            n_chips = st.slider("Chip count", min_value=100, max_value=2000, value=500, step=100)
            preset = st.selectbox("Preset", list(PRESETS.keys()))
            seed = st.number_input("Seed", value=42, step=1)
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader("Upload Data", type=["csv", "parquet", "pq"])
            n_chips = 0
            preset = "realistic_mixed"
            seed = 42

    with col_config2:
        st.subheader("Algorithm Parameters")
        st.markdown("---")
        pat_sigma = st.slider("PAT Sigma (n_sigma)", min_value=3.0, max_value=8.0, value=6.0, step=0.1)
        alpha = st.slider("Conformal Alpha", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
        ea_val = st.slider("Activation Energy (Ea) [eV]", min_value=0.6, max_value=1.1, value=0.7, step=0.05)
        mission_life = st.slider("Mission Life Requirement [years]", min_value=5, max_value=20, value=15, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        run_button = st.button("⚡ RUN SCREENING", type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("ℹ️ How this works (The 8-Stage Pipeline)"):
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

else:
    run_button = False


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
    # FIX: reference alpha safely in case it's missing (though it shouldn't be)
    _alpha = st.session_state.get('alpha_used', 0.05)
    lower = chip_row.get(f"pred_lower_{int((1-_alpha)*100)}", 0)
    upper = chip_row.get(f"pred_upper_{int((1-_alpha)*100)}", 0)
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
    
    indices = np.argsort(np.abs(shap_vals))
    top_indices = indices[-8:]
    
    x_labels = []
    y_vals = []
    text_vals = []
    
    for idx in top_indices:
        x_labels.append(f"{feat_names[idx]}<br>({feat_vals[idx]:.2f})")
        y_vals.append(shap_vals[idx])
        text_vals.append(f"{shap_vals[idx]:+.2f}")
        
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
        waterfallgap=0.3,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9")
    )
    return fig


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if run_button:
    loader_html = """
    <style>
    @keyframes pulseGlow {
      0%, 100% { opacity: 0.3; transform: scale(0.98); box-shadow: none; border-color: #1e293b; }
      50% { opacity: 1; transform: scale(1.05); box-shadow: 0 0 25px var(--stage-color); border-color: var(--stage-color); }
    }
    .pipeline-loader {
      display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; align-items: center; margin: 40px 0 60px 0;
    }
    .p-stage {
      display: flex; align-items: center; background: #0f172a; border: 2px solid #1e293b; border-radius: 8px; 
      padding: 10px; width: 220px; opacity: 0.3; animation: pulseGlow 3s infinite;
    }
    .p-stage.s1 { --stage-color: #94a3b8; animation-delay: 0.0s; }
    .p-stage.s2 { --stage-color: #facc15; animation-delay: 0.375s; }
    .p-stage.s3 { --stage-color: #f97316; animation-delay: 0.75s; }
    .p-stage.s4 { --stage-color: #c084fc; animation-delay: 1.125s; }
    .p-stage.s5 { --stage-color: #2dd4bf; animation-delay: 1.5s; }
    .p-stage.s6 { --stage-color: #60a5fa; animation-delay: 1.875s; }
    .p-stage.s7 { --stage-color: #38bdf8; animation-delay: 2.25s; }
    .p-stage.s8 { --stage-color: #4ade80; animation-delay: 2.625s; }
    
    .p-stage .num {
      background: var(--stage-color); color: #0a0f1e; font-weight: bold; font-family: 'Rajdhani', sans-serif; font-size: 1.3rem;
      width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; border-radius: 6px; margin-right: 12px; flex-shrink: 0;
    }
    .p-stage .desc { font-size: 0.75rem; color: #cbd5e1; line-height: 1.4; }
    .p-stage .desc b { color: #f8fafc; font-size: 0.95rem; font-family: 'Rajdhani', sans-serif; letter-spacing: 0.5px; }
    .p-arrow { color: #334155; font-size: 1.5rem; }
    
    .loader-title { text-align: center; font-family: 'Rajdhani', sans-serif; color: #f8fafc; margin-bottom: 20px; font-size: 1.5rem; letter-spacing: 2px;}
    .loader-title span { color: #f59e0b; }
    </style>
    
    <div class="loader-title">PROCESSING DATA THROUGH <span>AGNIPARIKSHA</span> PIPELINE</div>
    <div class="pipeline-loader">
      <div class="p-stage s1"><div class="num">1</div><div class="desc"><b>INGEST</b><br>Burn-in test logs</div></div><div class="p-arrow">➔</div>
      <div class="p-stage s2"><div class="num">2</div><div class="desc"><b>PAT SCREEN</b><br>Robust median ± 6σ</div></div><div class="p-arrow">➔</div>
      <div class="p-stage s3"><div class="num">3</div><div class="desc"><b>MULTIVARIATE</b><br>Mahalanobis MCD</div></div><div class="p-arrow">➔</div>
      <div class="p-stage s4"><div class="num">4</div><div class="desc"><b>ARRHENIUS FIT</b><br>Physics degradation</div></div><div class="p-arrow">➔</div>
      <div class="p-stage s5"><div class="num">5</div><div class="desc"><b>ML FORECAST</b><br>Predicts 168h value</div></div><div class="p-arrow">➔</div>
      <div class="p-stage s6"><div class="num">6</div><div class="desc"><b>CONFORMAL</b><br>95% upper bound</div></div><div class="p-arrow">➔</div>
      <div class="p-stage s7"><div class="num">7</div><div class="desc"><b>SHAP EXPLAIN</b><br>Per-chip attribution</div></div><div class="p-arrow">➔</div>
      <div class="p-stage s8"><div class="num">8</div><div class="desc"><b>DECISION</b><br>Aerospace pass/flag</div></div>
    </div>
    """
    
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(loader_html, unsafe_allow_html=True)
        
    try:
        if data_source == "Generate synthetic lot":
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
        
        df, mapie_model, eval_report = run_pipeline(df_long, alpha=alpha, model_type="rf")
        
        st.session_state["df_res"] = df
        st.session_state["mapie_model"] = mapie_model
        st.session_state["eval_report"] = eval_report
        st.session_state["run_success"] = True
        st.session_state["alpha_used"] = alpha
        st.rerun()
            
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
    alpha_used = st.session_state.get("alpha_used", 0.05)
    
    # -----------------------------------------------------------------------
    # Section 1: KPI Strip
    # -----------------------------------------------------------------------
    n_total = len(df)
    n_flagged = (df["verdict"] == "FLAG").sum()
    n_passed = n_total - n_flagged
    flag_rate = (n_flagged / n_total) * 100
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chips Screened", f"{n_total}", delta=None, border=True)
    k2.metric("Passed", f"{n_passed}", delta=None, border=True)
    k3.metric("Flagged", f"{n_flagged}", delta=None, border=True)
    k4.metric("Flag Rate", f"{flag_rate:.1f} %", delta=None, border=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # -----------------------------------------------------------------------
    # Tabs
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Screening Table", "Chip Analysis", "Performance", "Export"
    ])
    
    with tab1:
        st.subheader("Overview")
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.markdown("##### Rules Triggered")
            rule_counts = {
                "PAT Outlier": df["pat_flag"].sum(),
                "Mahalanobis": df["mv_flag"].sum(),
                "Conformal": df["conformal_flag"].sum(),
                "Lifetime": df["lifetime_flag"].sum(),
            }
            fig_rules = go.Figure(data=[
                go.Bar(
                    name="Fired", 
                    x=list(rule_counts.keys()), 
                    y=list(rule_counts.values()),
                    marker_color=['#3b82f6', '#22c55e', '#f59e0b', '#ef4444']
                )
            ])
            fig_rules.update_layout(
                yaxis_title="Number of Chips",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_rules, use_container_width=True)
            
        with c_right:
            st.markdown("##### The Blind Spot")
            timepoints = [0, 24, 96, 168]
            leakage_cols = [f"i_leak_ua_{t}h" for t in timepoints]
            
            fig_traj = go.Figure()
            
            x_grey = []
            y_grey = []
            for idx in df.index:
                chip_id = df.loc[idx, "chip_id"]
                if DEMO_CHIP_A_ID in chip_id or DEMO_CHIP_B_ID in chip_id:
                    continue
                x_grey.extend(timepoints + [None])
                y_grey.extend(df.loc[idx, leakage_cols].values.tolist() + [None])
                
            fig_traj.add_trace(go.Scatter(
                x=x_grey, y=y_grey, 
                mode="lines", 
                line=dict(color="rgba(148, 163, 184, 0.1)"), 
                name="Other Chips",
                hoverinfo="skip"
            ))
            
            demo_a = df[df["chip_id"].str.contains(DEMO_CHIP_A_ID)]
            if not demo_a.empty:
                a_row = demo_a.iloc[0]
                fig_traj.add_trace(go.Scatter(
                    x=timepoints, y=a_row[leakage_cols],
                    mode="lines+markers",
                    line=dict(color="#22c55e", width=3),
                    name="Healthy Chip"
                ))
                
            demo_b = df[df["chip_id"].str.contains(DEMO_CHIP_B_ID)]
            if not demo_b.empty:
                b_row = demo_b.iloc[0]
                fig_traj.add_trace(go.Scatter(
                    x=timepoints, y=b_row[leakage_cols],
                    mode="lines+markers",
                    line=dict(color="#f59e0b", width=3),
                    name="Latent Defect"
                ))
                
            spec_max = SPEC_LIMITS[COL_I_LEAK]["max"]
            fig_traj.add_hline(
                y=spec_max, line_dash="dash", line_color="#ef4444", 
                annotation_text=f"Spec Limit ({spec_max} µA)"
            )
            
            fig_traj.update_layout(
                xaxis_title="Burn-In Time [hours]",
                yaxis_title="Leakage Current [µA]",
                hovermode="closest",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_traj, use_container_width=True)

    with tab2:
        st.subheader("Screening Table")
        filter_val = st.radio("Filter Verdict:", ["ALL", "FLAG", "PASS"], horizontal=True, label_visibility="collapsed")
        
        df_disp = df.copy()
        if filter_val != "ALL":
            df_disp = df_disp[df_disp["verdict"] == filter_val]
            
        df_disp["Projected Life"] = df_disp.apply(
            lambda r: f"{r['life_years_ea07']:.1f} - {r['life_years_ea10']:.1f} yrs", axis=1
        )
        conf_level = int((1-alpha_used)*100)
        df_disp["168h Forecast 95% CI"] = df_disp.apply(
            lambda r: f"[{r[f'pred_lower_{conf_level}']:.1f}, {r[f'pred_upper_{conf_level}']:.1f}] µA", axis=1
        )
        df_disp["Mahalanobis D²"] = df_disp["mahalanobis_d2"].round(1)
        
        cols_to_show = ["chip_id", "lot_id", "verdict", "risk_score", "flag_reasons", "168h Forecast 95% CI", "Projected Life", "Mahalanobis D²"]
        
        def color_verdict(val):
            color = '#ef4444' if val == 'FLAG' else '#22c55e'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(
            df_disp[cols_to_show].style.map(color_verdict, subset=['verdict']),
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.subheader("Chip Analysis")
        selected_chip = st.selectbox("Select a chip to inspect:", df["chip_id"].unique(), label_visibility="collapsed")
        chip_data = df[df["chip_id"] == selected_chip].iloc[0]
        
        ca1, ca2 = st.columns(2)
        with ca1:
            st.markdown("##### Trajectory & Forecast")
            fig_drill = go.Figure()
            
            fig_drill.add_trace(go.Scatter(
                x=timepoints, y=chip_data[leakage_cols],
                mode="markers+lines", marker=dict(size=10, color="#3b82f6"),
                name="Measured [µA]"
            ))
            
            t_curve = np.linspace(0, 168, 100)
            p0 = chip_data[f"p0_fitted_{COL_I_LEAK}"]
            a = chip_data[f"drift_rate_a_{COL_I_LEAK}"]
            y_curve = p0 * np.exp(a * t_curve)
            
            fig_drill.add_trace(go.Scatter(
                x=t_curve, y=y_curve,
                mode="lines", line=dict(color="rgba(59, 130, 246, 0.4)", dash="dot"),
                name="Arrhenius Fit"
            ))
            
            pred = chip_data["pred_168h"]
            conf_level = int((1-alpha_used)*100)
            upper = chip_data[f"pred_upper_{conf_level}"]
            lower = chip_data[f"pred_lower_{conf_level}"]
            
            fig_drill.add_trace(go.Scatter(
                x=[168], y=[pred],
                mode="markers", marker=dict(size=12, color="#c084fc", symbol="diamond"),
                name="ML Forecast [µA]",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[upper - pred],
                    arrayminus=[pred - lower],
                    color="#c084fc"
                )
            ))
            
            spec_max = SPEC_LIMITS[COL_I_LEAK]["max"]
            fig_drill.add_hline(y=spec_max, line_dash="dash", line_color="#ef4444", annotation_text="Spec Limit")
            fig_drill.update_layout(
                xaxis_title="Time [h]", yaxis_title="Leakage [µA]",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"), margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_drill, use_container_width=True)
            
        with ca2:
            st.markdown("##### Lifetime Assessment")
            
            st.info(f"**Projected life:** {chip_data['life_years_ea07']:.1f} – {chip_data['life_years_ea10']:.1f} years\n\n"
                    f"*Range reflects activation-energy uncertainty, which we do not measure per chip.*")
                    
            st.success(f"**Rank in lot:** {chip_data['life_rank_in_lot']:.2f}× median\n\n"
                       f"*Ea-independent, the reliable comparison metric.*")
                       
            st.warning(f"**168h Forecast 95% Interval:** [{lower:.1f}, {upper:.1f}] µA\n\n"
                       f"*Covers regression error only (ML uncertainty), NOT physical Ea uncertainty.*")
                       
            v_color = "red" if chip_data['verdict'] == "FLAG" else "green"
            st.markdown(f"**Verdict:** <span style='color:{v_color}; font-size:1.2em; font-weight:bold;'>{chip_data['verdict']}</span>", unsafe_allow_html=True)
            if chip_data['verdict'] == "FLAG":
                st.error(f"**Triggered:** {', '.join(chip_data['flag_reasons'])}")

        st.markdown("##### Model Explanation (SHAP)")
        from src.forecast import build_features
        x_row = build_features(df[df["chip_id"] == selected_chip])
        explanation = explain_chip(model, x_row, list(x_row.columns))
        
        fig_shap = build_plotly_waterfall(explanation)
        st.plotly_chart(fig_shap, use_container_width=True)

    with tab4:
        st.subheader("Performance vs Baseline")
        if report is not None:
            rc1, rc2 = st.columns(2)
            with rc1:
                fig_recall = go.Figure(data=[
                    go.Bar(name="Latent-Defect Recall", x=["Conventional PAT", "AGNIPARIKSHA"], 
                           y=[report.pat_recall_latent * 100, report.full_recall_latent * 100],
                           marker_color=["#94a3b8", "#22c55e"])
                ])
                fig_recall.update_layout(
                    title="Recall on Latent Defects [%]", yaxis=dict(range=[0, 105]),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f1f5f9")
                )
                st.plotly_chart(fig_recall, use_container_width=True)
                
            with rc2:
                conf_level = int((1-alpha_used)*100)
                st.markdown(f"**Conformal Coverage:** {report.conformal_coverage * 100:.1f} % *(Nominal: {conf_level} %)*")
                st.markdown(f"**F1 Score (Pipeline):** {report.full_f1:.3f}")
                st.markdown(f"**F1 Score (PAT):** {report.pat_f1:.3f}")
                
                st.markdown("##### Pipeline Confusion Matrix")
                st.dataframe(pd.DataFrame(report.full_conf_matrix, 
                                          columns=["Pred PASS", "Pred FLAG"], 
                                          index=["True PASS", "True DEFECT"]))
        else:
            st.info("Performance report is not available for uploaded data without ground truth labels.")

    with tab5:
        st.subheader("Export Actions")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("#### Lot Data")
            flagged_df = df[df["verdict"] == "FLAG"]
            csv = flagged_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Flagged Chips CSV",
                data=csv,
                file_name='flagged_chips.csv',
                mime='text/csv',
                use_container_width=True
            )
            
        with ec2:
            st.markdown("#### Individual Reports")
            selected_chip_export = st.selectbox("Select chip for PDF report:", df[df["verdict"] == "FLAG"]["chip_id"].unique() if not df[df["verdict"] == "FLAG"].empty else ["None"])
            if selected_chip_export != "None":
                chip_data_export = df[df["chip_id"] == selected_chip_export].iloc[0]
                pdf_bytes = generate_rejection_pdf(chip_data_export)
                st.download_button(
                    label=f"📄 Download Rejection Report ({selected_chip_export})",
                    data=pdf_bytes,
                    file_name=f'rejection_{selected_chip_export}.pdf',
                    mime='application/pdf',
                    use_container_width=True
                )
            else:
                st.markdown("*(No flagged chips to report on)*")
