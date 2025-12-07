"""
Lane Change Assist Console
-------------------------------------------------
Real-time lane change decision support using ML.
"""

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go

# Page configuration (window title, icon, layout)
st.set_page_config(
    page_title="Lane Change Assist Console",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global custom styling for premium BMW-style dashboard UI
st.markdown(
    """
<style>
body { background-color: #050608; }

/* Main centered container */
.main-container {
    max-width: 1300px;
    margin: 0 auto;
    padding: 0 10px 40px 10px;
}

/* Card style sections */
.section-card {
    background: radial-gradient(circle at top left, #1f2733 0, #0b0d10 55%);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.04);
    box-shadow: 0 24px 40px rgba(0,0,0,0.65);
}

/* Header band */
.header-band {
    background: linear-gradient(135deg, rgba(0,176,255,0.18), rgba(0,255,200,0.10));
    border-radius: 999px;
    padding: 8px 20px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid rgba(255,255,255,0.12);
}

/* Typography setup */
h1, h2, h3, h4 {
    font-family: system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif;
    letter-spacing: .04em;
}
p, span, div, label {
    font-family: system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif;
}

/* Slider label styling */
.stSlider label, .stSelectSlider label, .stSelectbox label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #c5ccd8 !important;
}

/* Slider accent */
.stSlider > div[data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #00b0ff, #00ffc3);
}

/* Metrics grid */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 6px;
}
.metric-box {
    background: radial-gradient(circle at top, #222a35 0, #11141b 65%);
    border-radius: 10px;
    padding: 10px 12px;
    border: 1px solid rgba(255,255,255,0.08);
}
.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ba4b3;
}
.metric-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f5f7fb;
}

/* Status dots */
.status-dot {
    width: 9px; height: 9px; border-radius: 999px;
    display: inline-block; margin-right: 6px;
}
.status-ok { background: #00d68f; box-shadow:0 0 8px #00d68f;}
.status-warn { background: #ffb648; box-shadow:0 0 8px #ffb648;}
.status-bad { background: #ff4c5b; box-shadow:0 0 8px #ff4c5b;}

/* Progress bars */
.progress-shell {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 16px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #00ffc3, #00b0ff);
    transition: width 0.4s ease-out;
}

/* Prediction card look */
.prediction-panel {
    border-radius: 14px;
    padding: 18px 20px 16px 20px;
    border: 1px solid rgba(0,255,200,0.35);
    background: radial-gradient(circle at top left, rgba(0,255,200,0.16), rgba(0,0,0,0.85));
}
.pred-title { font-size: 0.88rem; text-transform:uppercase; letter-spacing:0.13em; color:#9ba4b3; margin-bottom:4px;}
.pred-main { font-size:1.5rem; font-weight:650; letter-spacing:0.18em; color:#f7fafc;}
.pred-caption { font-size:0.85rem; color:#c0c7d4; margin-top:6px;}

.dataframe tbody tr:nth-child(odd) { background:rgba(255,255,255,0.02); }
.dataframe tbody tr:nth-child(even) { background:rgba(0,0,0,0.0); }

/* Footer line */
.footer-line {
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 18px;
    padding-top: 10px;
    font-size: 0.8rem;
    color: #7c8597;
}
</style>
""",
    unsafe_allow_html=True,
)

# Header presentation
st.markdown(
    """
<div class="main-container">
  <div style="margin-top: 6px; margin-bottom: 16px;">
      <div class="header-band">
          <span style="width: 8px; height: 8px; border-radius:999px; background:#00ffc3; box-shadow:0 0 10px #00ffc3;"></span>
          <span style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.18em; color:#d0d7e4;">
              Lane Change Assist • Real-Time Console
          </span>
      </div>
      <h1 style="margin:10px 0 4px 0; color:#f5f7fb; font-size:1.9rem;">
          Lane Change Assist System
      </h1>
      <p style="margin:0; color:#9fa7b6; font-size:0.9rem;">
          Machine-learning based decision support for lane keeping and lane shift manoeuvres.
      </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Load trained ML model from file
@st.cache_resource
def load_model():
    try:
        return joblib.load("lane_model.pkl")
    except Exception as e:
        st.error("Trained model file 'lane_model.pkl' not found.")
        st.code(str(e))
        return None

model = load_model()
if model is None:
    st.stop()

# Sidebar controls (simulation + presets + summary)
with st.sidebar:
    st.markdown("#### Simulation Panel")
    st.caption("Configure how the lane-change assistant is exercised inside lab conditions.")

    realtime = st.checkbox("Continuous refresh", value=False)

    if realtime:
        rate = st.slider("Refresh rate (evaluations / second)", 1, 10, 3)
    else:
        rate = None

    st.markdown("---")
    st.markdown("#### Scenario Presets")

    preset = st.selectbox(
        "Select scenario",
        [
            "Custom configuration",
            "Highway – stable cruise",
            "Highway – right overtake",
            "Urban – congestion on left",
            "Curved segment – left bend",
            "Curved segment – right bend",
        ],
    )

    st.markdown("---")
    st.markdown("#### System Summary")
    st.markdown(
        """
- Model: Random Forest classifier  
- Inputs: speed, steering, lane clearance, road curvature  
- Output: lane hold / lane shift recommendation  
"""
    )

# Default slider values based on preset selection
default_speed = 80
default_left = 1.5
default_right = 1.5
default_steer = 0
default_curve = 0

if preset == "Highway – stable cruise":
    default_speed, default_left, default_right = 100, 1.9, 1.8
elif preset == "Highway – right overtake":
    default_speed, default_left, default_right, default_steer = 110, 1.2, 2.3, 12
elif preset == "Urban – congestion on left":
    default_speed, default_left, default_right = 55, 0.6, 2.0
elif preset == "Curved segment – left bend":
    default_speed, default_left, default_right, default_curve = 75, 1.7, 1.1, -18
elif preset == "Curved segment – right bend":
    default_speed, default_left, default_right, default_curve = 75, 1.1, 1.7, 18

# Main layout columns
col_controls, col_status, col_pred = st.columns([2.1, 1.4, 1.8])

# VEHICLE & ENVIRONMENT CONTROL PANEL
with col_controls:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Vehicle Dynamics")

    c1, c2 = st.columns(2)
    with c1:
        speed = st.slider("Speed (km/h)", 30, 150, int(default_speed))
        dist_left = st.slider("Left lane clearance (m)", 0.2, 4.0, float(default_left), 0.1)
    with c2:
        steering = st.slider("Steering angle (°)", -40, 40, int(default_steer))
        dist_right = st.slider("Right lane clearance (m)", 0.2, 4.0, float(default_right), 0.1)

    st.markdown("---")
    st.markdown("##### Environment Model")

    c3, c4 = st.columns(2)
    with c3:
        road_curve = st.slider("Road curvature (°)", -30, 30, int(default_curve))
    with c4:
        visibility = st.select_slider("Visibility", ["Poor", "Average", "Good"], "Good")
        traffic = st.select_slider("Traffic density", ["Low", "Medium", "High"], "Medium")

    st.markdown("</div>", unsafe_allow_html=True)

# LIVE VEHICLE STATUS
with col_status:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##### Live Vehicle State")

    left_safe = dist_left > 0.8
    right_safe = dist_right > 0.8
    speed_safe = speed <= 100

    st.markdown(
        """
    <div style="font-size:0.82rem; text-transform:uppercase; letter-spacing:0.12em; color:#9ba4b3; margin-bottom:4px;">
        Lane clearance
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)

    left_color = "status-ok" if left_safe else "status-warn" if dist_left > 0.5 else "status-bad"
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label"><span class="status-dot {left_color}"></span>Left clearance</div>
            <div class="metric-value">{dist_left:.2f} m</div>
        </div>""",
        unsafe_allow_html=True,
    )

    right_color = "status-ok" if right_safe else "status-warn" if dist_right > 0.5 else "status-bad"
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label"><span class="status-dot {right_color}"></span>Right clearance</div>
            <div class="metric-value">{dist_right:.2f} m</div>
        </div>""",
        unsafe_allow_html=True,
    )

    spd_color = "status-ok" if speed_safe else "status-warn" if speed <= 120 else "status-bad"
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label"><span class="status-dot {spd_color}"></span>Longitudinal speed</div>
            <div class="metric-value">{speed} km/h</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.82rem; text-transform:uppercase; letter-spacing:0.12em; color:#9ba4b3;'>Lane position – top view</div>", unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Right boundary", "Vehicle", "Left boundary"],
            x=[dist_right, 0.6, dist_left],
            orientation="h",
            marker_color=["#ff4c5b", "#1f2933", "#00b0ff"],
            text=[f"{dist_right:.2f} m", "", f"{dist_left:.2f} m"],
            textposition="outside"
        )
    )

    fig.update_layout(
        height=160,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(visible=False),
        plot_bgcolor="#050608",
        paper_bgcolor="#050608",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# PREDICTION & RISK ANALYSIS
with col_pred:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    features = np.array([[speed, steering, dist_left, dist_right, road_curve]])
    pred_raw = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(np.max(probabilities) * 100)

    if pred_raw == "KEEP_LANE":
        disp_label = "LANE HOLD"
        accent_color = "#00ffc3"
        description = "Maintain current lane; lateral position is normal."
    elif pred_raw == "CHANGE_LEFT":
        disp_label = "LANE SHIFT LEFT"
        accent_color = "#00b0ff"
        description = "Left lane offers improved clearance."
    else:
        disp_label = "LANE SHIFT RIGHT"
        accent_color = "#ff4c5b"
        description = "Right lane offers improved clearance."

    st.markdown(
        f"""
        <div class="prediction-panel" style="border-color:{accent_color};">
            <div class="pred-title">Decision engine</div>
            <div class="pred-main">{disp_label}</div>
            <div class="pred-caption">{description}</div>
            <div style="margin-top:14px;">
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.11em; color:#9ba4b3;">Confidence</div>
                <div class="progress-shell">
                    <div class="progress-fill" style="width:{confidence:.1f}%; background:linear-gradient(90deg, {accent_color}, #ffffff);"></div>
                </div>
                <div style="margin-top:4px; font-size:0.86rem; color:#dde3f0;">
                    {confidence:.1f}% model confidence for the recommended manoeuvre.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.82rem;text-transform:uppercase;letter-spacing:0.12em;color:#9ba4b3;'>Adaptive safety index</div>", unsafe_allow_html=True)

    risk_score = 0
    if dist_left < 0.8: risk_score += 22
    if dist_right < 0.8: risk_score += 22
    if abs(steering) > 25: risk_score += 18
    if speed > 100: risk_score += 15
    if visibility == "Average": risk_score += 6
    elif visibility == "Poor": risk_score += 14
    if traffic == "Medium": risk_score += 6
    elif traffic == "High": risk_score += 12
    risk_score = min(risk_score, 100)

    if risk_score < 35:
        risk_label, risk_color = "LOW", "#00ffc3"
    elif risk_score < 70:
        risk_label, risk_color = "MODERATE", "#ffb648"
    else:
        risk_label, risk_color = "ELEVATED", "#ff4c5b"

    st.markdown(
        f"""
        <div class="progress-shell" style="height:18px;">
            <div class="progress-fill" style="width:{risk_score}%; background:linear-gradient(90deg,#00ffc3,#ffb648,#ff4c5b);"></div>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:0.82rem; color:#a2abb9;">
            <span>Index: {risk_score:.0f} / 100</span>
            <span style="letter-spacing:0.16em; text-transform:uppercase; color:{risk_color};">{risk_label} RISK</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("<div style='font-size:0.82rem;text-transform:uppercase;letter-spacing:0.12em;color:#9ba4b3;'>Decision distribution</div>", unsafe_allow_html=True)

    classes = model.classes_
    probs = probabilities * 100

    for c, p in zip(classes, probs):
        label = "Lane hold" if c == "KEEP_LANE" else "Lane shift left" if c == "CHANGE_LEFT" else "Lane shift right"
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
                <div style="width:130px;font-size:0.78rem;text-transform:uppercase;color:#9ca5b6;">{label}</div>
                <div style="flex:1;">
                    <div class="progress-shell">
                        <div class="progress-fill" style="width:{p:.1f}%"></div>
                    </div>
                </div>
                <div style="width:48px;text-align:right;font-size:0.85rem;color:#e6ebf5;">{p:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# DECISION HISTORY + RISK TREND
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown("##### Recent evaluations")

if "history" not in st.session_state:
    st.session_state.history = []

history_entry = {
    "Time": datetime.now().strftime("%H:%M:%S"),
    "Decision": pred_raw,
    "Confidence (%)": round(confidence, 1),
    "SafetyIndex": risk_score,
    "Speed (km/h)": speed,
    "Left (m)": round(dist_left, 2),
    "Right (m)": round(dist_right, 2),
}

st.session_state.history.append(history_entry)
if len(st.session_state.history) > 20:
    st.session_state.history = st.session_state.history[-20:]

if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)

    def row_style(row):
        base = "background-color: rgba(255,255,255,0.00); color:#e0e4ef;"
        return [base] * len(row)

    st.dataframe(hist_df.style.apply(row_style, axis=1), hide_index=True, height=260)

    risk_vals = [h["SafetyIndex"] for h in st.session_state.history]
    fig_risk = go.Figure()
    fig_risk.add_trace(
        go.Scatter(
            x=list(range(len(risk_vals))),
            y=risk_vals,
            mode="lines+markers",
            line=dict(width=2.5, color="#ff4c5b"),
            fill="tozeroy",
            fillcolor="rgba(255,76,91,0.12)",
        )
    )

    fig_risk.update_layout(
        title="Safety index trajectory",
        xaxis_title="Evaluation index (latest on right)",
        yaxis_title="Safety index",
        height=230,
        margin=dict(l=0, r=0, t=40, b=10),
        plot_bgcolor="#050608",
        paper_bgcolor="#050608",
        font=dict(color="#d5d9e4"),
    )
    st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    """
<div class="footer-line">
    Lane Change Assist System – prototype console for lane-keeping / lane-shift decision analytics.
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div></div>", unsafe_allow_html=True)

# LIVE simulation loop
if realtime and rate:
    time.sleep(1 / rate)
    st.rerun()
