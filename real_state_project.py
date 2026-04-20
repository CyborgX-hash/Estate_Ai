import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

from core.state import AgentState
from core.validators import validate_input
from data.loader import load_data, clean_data
from models.predictor import train_model
from ai.rag import get_vectorstore
from ai.advisor import get_llm
from agent.graph import build_graph
from ui.styles import apply_styles

st.set_page_config(
    page_title="EstateAI",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply UI CSS styles
apply_styles()

# Initialize data and models
df_raw         = load_data()
df_clean       = clean_data(df_raw)
model, feature_columns, model_metrics = train_model(df_clean)
vectorstore    = get_vectorstore()
llm            = get_llm()
agent_app      = build_graph()

# Persist in session state
st.session_state["model"]           = model
st.session_state["feature_columns"] = feature_columns
st.session_state["vectorstore"]     = vectorstore
st.session_state["llm"]             = llm
st.session_state["df_raw"]          = df_raw

# Build UI Layout
with st.sidebar:
    st.markdown('<div class="sidebar-section">Model Performance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² Score",   f"{model_metrics['R² Score']:.4f}")
        st.metric("Train Rows", f"{model_metrics['Train Rows']:,}")
    with c2:
        st.metric("MAE",        f"₹{model_metrics['MAE']:,.0f}")
        st.metric("Test Rows",  f"{model_metrics['Test Rows']:,}")
    st.metric("RMSE", f"₹{model_metrics['RMSE']:,.0f}")

    st.markdown('<div class="sidebar-section">Agent Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-box">
        Input <span class="arrow">→</span> <span class="node">predict_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">rag_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">comps_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">advisor_node</span><br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="arrow">→</span> <span class="node">END</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Stack</div>', unsafe_allow_html=True)
    for pill in ["Random Forest","FAISS RAG","Flan-T5","LangGraph","Streamlit"]:
        st.markdown(f'<span class="pill">{pill}</span>', unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("EstateAI v2.0 · For informational use only")


st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">AI-Powered Real Estate Intelligence</div>
    <div class="hero-title">Estate<span>AI</span></div>
    <p class="hero-sub">
        Agentic property valuation &nbsp;·&nbsp; FAISS market retrieval &nbsp;·&nbsp; Structured investment advisory
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="section-heading">Property Details</div>
<div class="section-sub">Enter the property parameters to generate your advisory report</div>
<div class="gold-line"></div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown('<div class="input-group-title">Size & Layout</div>', unsafe_allow_html=True)
        carpet_area   = st.number_input("Carpet Area (sq ft)", min_value=100.0, max_value=50000.0, value=1000.0, step=50.0)
        num_rooms     = st.number_input("Number of Rooms",      min_value=1,     max_value=20,      value=3)
        num_bathrooms = st.number_input("Number of Bathrooms",  min_value=1,     max_value=15,      value=2)

with col2:
    with st.container(border=True):
        st.markdown('<div class="input-group-title">Financials</div>', unsafe_allow_html=True)
        estimated_value = st.number_input("Estimated Value (₹)", min_value=100_000.0, value=2_000_000.0, step=50_000.0, format="%.0f")
        tax_rate        = st.number_input("Property Tax Rate (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

with col3:
    with st.container(border=True):
        st.markdown('<div class="input-group-title">Time Period</div>', unsafe_allow_html=True)
        year  = st.number_input("Year",  min_value=1990, max_value=2030, value=2023)
        month = st.slider("Month", 1, 12, 6)

st.markdown("<br>", unsafe_allow_html=True)


_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_clicked = st.button("Analyse Property & Generate Report")


if run_clicked:

    input_data: Dict[str, Any] = {
        "property_tax_rate": tax_rate,
        "carpet_area":       carpet_area,
        "num_bathrooms":     num_bathrooms,
        "num_rooms":         num_rooms,
        "Estimated Value":   estimated_value,
        "Year":              year,
        "month":             month,
    }

    errors = validate_input(input_data)
    if errors:
        for e in errors:
            st.error(f"{e}")
        st.stop()

    with st.spinner("Agent running — predicting · retrieving · advising…"):
        try:
            initial_state: AgentState = {
                "input": input_data, "predicted_price": 0.0,
                "market_data": [], "comps": [], "final_advice": "",
                "model_metrics": model_metrics, "error": "",
            }
            result = agent_app.invoke(initial_state)
        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            st.stop()

    if result.get("error"):
        st.warning(f"Note: {result['error']}")

    price    = result["predicted_price"]
    val_diff = price - estimated_value
    diff_pct = (val_diff / estimated_value) * 100 if estimated_value else 0
    ppsf     = price / carpet_area if carpet_area else 0

    direction  = "▲" if val_diff > 0 else "▼"
    diff_color = "#2ECC71" if val_diff > 0 else "#E74C3C"
    diff_label = "above" if val_diff > 0 else "below"

    st.markdown(f"""
    <div class="price-hero">
        <div class="price-label">Predicted Market Price</div>
        <div class="price-value">₹{price:,.0f}</div>
        <div class="price-sub">
            {direction} &nbsp;
            <span style="color:{diff_color}; font-weight:600;">{abs(diff_pct):.1f}% {diff_label}</span>
            &nbsp; your estimate of ₹{estimated_value:,.0f}
            &nbsp;&nbsp;·&nbsp;&nbsp; ₹{ppsf:,.0f} per sq ft
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">Advisory Report</div>
    <div class="section-sub">AI-generated structured investment analysis across 4 sections</div>
    <div class="gold-line"></div>
    """, unsafe_allow_html=True)

    with st.expander("Section 1 — Property Summary & Valuation", expanded=True):
        st.markdown('<div class="report-section-title">Property Summary</div>', unsafe_allow_html=True)
        val_color = "#2ECC71" if val_diff > 0 else "#E74C3C"
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">₹{price:,.0f}</div>
                <div class="metric-label">Predicted Price</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">₹{ppsf:,.0f}</div>
                <div class="metric-label">Price / sq ft</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:{val_color}">{diff_pct:+.1f}%</div>
                <div class="metric-label">vs Your Estimate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{carpet_area:,.0f}</div>
                <div class="metric-label">Area (sq ft)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{num_rooms}</div>
                <div class="metric-label">Rooms</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{num_bathrooms}</div>
                <div class="metric-label">Bathrooms</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Section 2 — Market Intelligence (RAG)", expanded=True):
        st.markdown('<div class="report-section-title">Retrieved Market Insights</div>', unsafe_allow_html=True)
        rows_html = "".join(
            f'<div class="insight-row"><div class="insight-num">{i}</div>'
            f'<div class="insight-text">{ins}</div></div>'
            for i, ins in enumerate(result["market_data"], 1)
        )
        st.markdown(rows_html, unsafe_allow_html=True)

    with st.expander("Section 3 — Comparable Properties (Comps)", expanded=True):
        st.markdown('<div class="report-section-title">Similar Properties in Dataset</div>', unsafe_allow_html=True)
        comps = result.get("comps", [])
        if comps:
            st.dataframe(pd.DataFrame(comps), use_container_width=True, hide_index=True)
            comp_prices = []
            for c in comps:
                try:
                    comp_prices.append(float(c["Sale Price"].replace("₹","").replace(",","")))
                except: pass
            if comp_prices:
                avg   = np.mean(comp_prices)
                delta = price - avg
                lc, rc = st.columns(2)
                lc.metric("Average Comparable Price", f"₹{avg:,.0f}")
                rc.metric("Your Prediction vs Comps", f"₹{price:,.0f}", delta=f"₹{delta:+,.0f}")
        else:
            st.info("No comparable properties found for the given filters.")

    with st.expander("Section 4 — AI Investment Advice", expanded=True):
        st.markdown('<div class="report-section-title">Structured Advisory</div>', unsafe_allow_html=True)
        advice_html = result['final_advice'].replace("\n", "<br>")
        st.markdown(f'<div class="advice-card">{advice_html}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <div class="disclaimer-title">Legal & Financial Disclaimer</div>
        <div class="disclaimer-text">
            This report is produced by an AI system for <strong>informational and educational purposes only</strong>.
            It does <strong>not</strong> constitute professional financial, investment, or legal advice.
            Predictions are based on historical training data and do not guarantee future results.
            Real estate markets are subject to volatility, regulatory changes, and macro-economic risks.<br><br>
            Always consult a <strong>SEBI-registered investment advisor</strong>, a <strong>licensed property valuer</strong>,
            and a qualified <strong>legal professional</strong> before making any real estate investment decision.
            The creators of EstateAI assume no liability for decisions made using this report.
        </div>
    </div>
    """, unsafe_allow_html=True)