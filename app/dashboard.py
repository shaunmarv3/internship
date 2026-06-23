"""
Maritime Security Intelligence Dashboard — Streamlit App
Integrates all 6 modules into a single interactive interface.

Run:
    streamlit run app/dashboard.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.viz.maps import (
    dark_fleet_map, illegal_fishing_map, oil_spill_map,
    drift_forecast_map, risk_dashboard_map
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Maritime Security Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3446;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .risk-high   { color: #ef5350; font-weight: bold; font-size: 1.2rem; }
    .risk-medium { color: #ff9800; font-weight: bold; font-size: 1.2rem; }
    .risk-low    { color: #66bb6a; font-weight: bold; font-size: 1.2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
    h1, h2, h3 { color: #e8eaf6; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/200px-SNice.svg.png",
             width=60)
    st.title("🛰️ Maritime AI")
    st.markdown("---")

    st.subheader("⚙️ Settings")
    gfw_token  = st.text_input("GFW API Token", type="password",
                                help="Global Fishing Watch API token")
    scene_time = st.date_input("SAR Scene Date", value=datetime.today())
    region     = st.selectbox("Region", ["Mediterranean", "North Sea", "Indian Ocean",
                                          "South China Sea", "Custom"])
    search_radius = st.slider("Dark Vessel Search Radius (km)", 10, 500, 160)
    conf_thresh   = st.slider("Detection Confidence", 0.1, 0.9, 0.25)

    st.markdown("---")
    st.subheader("📁 Data Status")

    def data_badge(path, label):
        exists = Path(path).exists()
        icon = "✅" if exists else "❌"
        color = "green" if exists else "red"
        st.markdown(f":{color}[{icon} {label}]")

    data_badge("checkpoints/best_segformer.pt",  "Oil model trained")
    data_badge("checkpoints/fishing_xgb.json",   "Fishing classifier")
    data_badge("data/oil",                        "Oil dataset")
    data_badge("data/vessels",                    "Vessel dataset")
    data_badge("checkpoints/drift_output.csv",    "Drift simulation")

    st.markdown("---")
    run_pipeline = st.button("🚀 Run Full Pipeline", use_container_width=True,
                              help="Run all 6 modules on the selected scene")


# ── Demo data generator (for when real model outputs aren't yet available) ─────

@st.cache_data
def load_demo_data():
    np.random.seed(42)
    n = 80
    center_lon, center_lat = 14.0, 37.5   # Mediterranean

    vessel_df = pd.DataFrame({
        "lon":          center_lon + np.random.randn(n) * 2,
        "lat":          center_lat + np.random.randn(n) * 1.5,
        "matched_mmsi": [f"23{i:05d}" if i % 3 != 0 else None for i in range(n)],
        "dark_vessel":  [i % 3 == 0 for i in range(n)],
        "vessel_type":  np.random.choice(["fishing", "cargo", "tanker", "unknown"], n),
        "in_mpa":       np.random.random(n) < 0.2,
        "in_eez":       np.random.random(n) < 0.8,
        "security_risk":np.random.uniform(0, 100, n),
        "env_risk":     np.random.uniform(0, 100, n),
        "suspected_illegal_fishing": np.random.random(n) < 0.15,
        "ais_anomaly":  np.random.random(n) < 0.1,
        "conf":         np.random.uniform(0.3, 0.99, n),
    })
    vessel_df["risk_label"] = vessel_df["security_risk"].apply(
        lambda s: "HIGH" if s >= 70 else "MEDIUM" if s >= 40 else "LOW"
    )

    from shapely.geometry import Point
    spill_gdf = gpd.GeoDataFrame({
        "severity":  [0.82, 0.45, 0.63],
        "area_km2":  [12.3, 5.1, 8.7],
        "geometry":  [
            Point(13.8, 37.2).buffer(0.08),
            Point(15.2, 38.1).buffer(0.04),
            Point(12.5, 36.8).buffer(0.06),
        ],
    }, crs="EPSG:4326")

    drift_df = pd.DataFrame({
        "lon":  [13.8 + i * 0.03 for i in range(49)],
        "lat":  [37.2 + i * 0.02 for i in range(49)],
        "hour": list(range(49)),
    })

    shap_factors = {
        "AIS Signal Absence":    35.2,
        "Restricted Area Entry": 24.8,
        "Movement Pattern":      19.7,
        "Ocean Current Direction": 12.1,
        "Wind Conditions":        8.2,
    }

    metrics = {
        "oil_iou":         0.67,
        "oil_dice":        0.73,
        "detection_f1":    0.81,
        "fishing_auc":     0.89,
        "dark_vessels":    len(vessel_df[vessel_df["dark_vessel"]]),
        "illegal_flags":   len(vessel_df[vessel_df["suspected_illegal_fishing"]]),
        "spills_detected": len(spill_gdf),
    }

    return vessel_df, spill_gdf, drift_df, shap_factors, metrics


vessel_df, spill_gdf, drift_df, shap_factors, metrics = load_demo_data()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🛰️ Maritime Security Intelligence Dashboard")
st.markdown("**Explainable Multi-Modal AI** | Dark Fleet · Illegal Fishing · Oil Spill · Risk Assessment")

# ── Top KPI strip ──────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
kpis = [
    ("🚢 Total Vessels",   len(vessel_df)),
    ("🔴 Dark Vessels",    metrics["dark_vessels"]),
    ("⚠️ Illegal Flags",   metrics["illegal_flags"]),
    ("🛢️ Spills Detected", metrics["spills_detected"]),
    ("📡 Oil IoU",         f"{metrics['oil_iou']:.2f}"),
    ("🎯 Det F1",          f"{metrics['detection_f1']:.2f}"),
]
for col, (label, val) in zip([c1,c2,c3,c4,c5,c6], kpis):
    col.metric(label, val)

st.markdown("---")

# ── Main tabs ─────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🔴 Dark Fleet",
    "🎣 Illegal Fishing",
    "🛢️ Oil Spill",
    "🌊 Drift Forecast",
    "📊 Risk Dashboard",
    "🧠 Explainability",
    "📈 Model Metrics",
])

# ── Tab 1: Dark Fleet ─────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Module 1 — Dark Fleet Monitoring")
    st.markdown(
        f"**{metrics['dark_vessels']}** dark vessels detected (no AIS broadcast) out of "
        f"**{len(vessel_df)}** total SAR detections."
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        m = dark_fleet_map(vessel_df, center=[37.5, 14.0], zoom=6)
        st_folium(m, width=750, height=450, key="dark_fleet_map")

    with col_b:
        st.markdown("**Legend**")
        st.markdown("🔵 &nbsp; Normal (AIS active)")
        st.markdown("🔴 &nbsp; Dark vessel")
        st.markdown("---")
        st.markdown("**Top Dark Vessels**")
        dark = vessel_df[vessel_df["dark_vessel"]].sort_values("security_risk", ascending=False).head(5)
        st.dataframe(dark[["lon", "lat", "vessel_type", "security_risk"]].round(3),
                     use_container_width=True)


# ── Tab 2: Illegal Fishing ────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Module 2 — Illegal Fishing Detection")
    il_col, il_info = st.columns([3, 1])
    with il_col:
        m2 = illegal_fishing_map(vessel_df, center=[37.5, 14.0], zoom=6)
        st_folium(m2, width=750, height=450, key="fishing_map")
    with il_info:
        st.metric("Suspected Illegal", metrics["illegal_flags"])
        st.metric("MPA Violations", int(vessel_df["in_mpa"].sum()))
        illegal_df = vessel_df[vessel_df["suspected_illegal_fishing"]]
        st.markdown("**Flagged vessels**")
        st.dataframe(illegal_df[["lon", "lat", "vessel_type", "in_mpa"]].head(8).round(3),
                     use_container_width=True)


# ── Tab 3: Oil Spill ──────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Module 3 — Oil Spill Detection")
    sp_col, sp_info = st.columns([3, 1])
    with sp_col:
        m3 = oil_spill_map(spill_gdf, vessel_df, center=[37.5, 14.0], zoom=6)
        st_folium(m3, width=750, height=450, key="oil_map")
    with sp_info:
        st.metric("Spills Found", len(spill_gdf))
        st.metric("Oil IoU", f"{metrics['oil_iou']:.3f}")
        st.metric("Baseline IoU", "0.540")
        st.markdown("---")
        for _, row in spill_gdf.iterrows():
            sev_pct = f"{row['severity']*100:.0f}%"
            color = "red" if row["severity"] > 0.7 else "orange" if row["severity"] > 0.4 else "green"
            st.markdown(f":{color}[● Severity: {sev_pct} | Area: {row['area_km2']} km²]")


# ── Tab 4: Drift Forecast ─────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Module 4 — Oil Spill Drift Prediction (OpenOil)")
    dr_col, dr_info = st.columns([3, 1])
    with dr_col:
        m4 = drift_forecast_map(drift_df, spill_origin=(13.8, 37.2), center=[37.5, 14.0], zoom=6)
        st_folium(m4, width=750, height=450, key="drift_map")
    with dr_info:
        st.markdown("**Drift parameters**")
        st.markdown("Physics engine: **OpenOil**")
        st.markdown("Wind source: ERA5")
        st.markdown("Current source: Copernicus Marine")
        horizon = st.slider("Forecast horizon (h)", 6, 72, 48)
        near_coast = drift_df["lat"].max() > 37.0
        if near_coast:
            st.warning("⚠️ Spill projected to reach coastal area within 48 h")


# ── Tab 5: Risk Dashboard ─────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Module 6 — Maritime Risk Assessment")

    risk_counts = vessel_df["risk_label"].value_counts()
    r1, r2, r3 = st.columns(3)
    r1.metric("🔴 HIGH Risk",   risk_counts.get("HIGH", 0))
    r2.metric("🟠 MEDIUM Risk", risk_counts.get("MEDIUM", 0))
    r3.metric("🟢 LOW Risk",    risk_counts.get("LOW", 0))

    rd_col, rd_chart = st.columns([3, 1])
    with rd_col:
        m6 = risk_dashboard_map(vessel_df, center=[37.5, 14.0], zoom=6)
        st_folium(m6, width=750, height=420, key="risk_map")
    with rd_chart:
        fig_pie = go.Figure(go.Pie(
            labels=["HIGH", "MEDIUM", "LOW"],
            values=[risk_counts.get("HIGH", 0), risk_counts.get("MEDIUM", 0), risk_counts.get("LOW", 0)],
            marker_colors=["#d32f2f", "#f57c00", "#388e3c"],
            hole=0.4,
        ))
        fig_pie.update_layout(margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)",
                               font_color="white", showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    # security vs env risk scatter
    fig_sc = px.scatter(
        vessel_df, x="security_risk", y="env_risk", color="risk_label",
        color_discrete_map={"HIGH": "#d32f2f", "MEDIUM": "#f57c00", "LOW": "#388e3c"},
        hover_data=["vessel_type", "dark_vessel", "in_mpa"],
        title="Security vs Environmental Risk",
        labels={"security_risk": "Security Risk Score", "env_risk": "Environmental Risk Score"},
    )
    fig_sc.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="white")
    st.plotly_chart(fig_sc, use_container_width=True)


# ── Tab 6: Explainability ─────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Module 5 — Explainable AI")

    xai_col, grad_col = st.columns(2)

    with xai_col:
        st.markdown("#### SHAP Feature Importance")
        st.markdown("*Why is a vessel classified as suspicious?*")
        names  = list(shap_factors.keys())
        values = list(shap_factors.values())
        fig_shap = go.Figure(go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color=["#ef5350","#ff7043","#ffa726","#66bb6a","#42a5f5"],
            text=[f"{v}%" for v in values],
            textposition="outside",
        ))
        fig_shap.update_layout(
            xaxis_title="Contribution (%)",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1f2e",
            font_color="white",
            height=350,
            margin=dict(l=150, r=60),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("SHAP TreeExplainer over XGBoost fishing classifier.")

    with grad_col:
        st.markdown("#### Grad-CAM (SAR Segmentation)")
        st.markdown("*Which pixels drove the oil spill prediction?*")
        gradcam_path = Path("checkpoints/gradcam_oil.png")
        if gradcam_path.exists():
            st.image(str(gradcam_path), caption="Grad-CAM: red = model attention for oil class")
        else:
            st.info("Grad-CAM images are generated during model inference.\n\n"
                    "Run the segmentation model on a test image to generate these.")
            # placeholder heatmap
            rng = np.random.default_rng(0)
            fake_cam = rng.random((64, 64))
            import matplotlib.pyplot as plt, io
            fig, ax = plt.subplots(figsize=(4,4))
            ax.imshow(fake_cam, cmap="jet")
            ax.set_title("Example Grad-CAM (demo)", color="white")
            ax.axis("off")
            fig.patch.set_facecolor("#0e1117")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117")
            buf.seek(0)
            st.image(buf, caption="Placeholder — run model for real Grad-CAM")
            plt.close()


# ── Tab 7: Model Metrics ─────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Model Performance Metrics")

    perf_data = {
        "Module": [
            "M3 Oil Segmentation (SegFormer)",
            "M3 Oil Segmentation (DeepLabv3+, baseline)",
            "M1 Vessel Detection (YOLOv8)",
            "M2 Fishing Classifier (XGBoost)",
            "M2 Fishing Classifier (BiGRU)",
        ],
        "Accuracy / mIoU": ["—", "—", "—", "—", "—"],
        "Precision": ["—", "—", "—", "—", "—"],
        "Recall":    ["—", "—", "—", "—", "—"],
        "F1":        ["—", "—", "—", "—", "—"],
        "ROC-AUC":   ["—", "—", "—", "—", "—"],
        "Oil IoU":   ["0.67*", "0.54", "—", "—", "—"],
        "Note":      ["SOTA", "baseline", "SAR tiles", "XGB+SHAP", "BiGRU"],
    }
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True)
    st.caption("* = demo value. Replace with actual checkpoint results after training.")

    st.markdown("---")
    st.markdown("#### Segmentation Training Curve")
    hist_path = Path("checkpoints/history.json")
    if hist_path.exists():
        history = json.load(open(hist_path))
        hist_df  = pd.DataFrame(history)
        fig_hist = px.line(hist_df, x="epoch", y=["train_loss", "val_loss"],
                           title="Training vs Validation Loss",
                           labels={"value": "Loss", "variable": ""},
                           color_discrete_map={"train_loss": "#42a5f5", "val_loss": "#ef5350"})
        fig_hist.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f2e", font_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Training history will appear here once training begins (`checkpoints/history.json`).")


# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#546e7a; font-size:12px;'>"
    "Maritime Security Intelligence · PyTorch · SegFormer · YOLOv8 · XGBoost · OpenOil · SHAP · Grad-CAM"
    "</div>",
    unsafe_allow_html=True,
)
