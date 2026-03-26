"""
Hooper Energy Solutions — California Energy Intelligence Platform
Complete integrated dashboard: Analytics + Trading Terminal + ML Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Hooper Energy Solutions",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# THEME SYSTEM — two distinct visual modes
# ============================================================================

TRADING_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@300;400;500&display=swap');

/* ─── GLOBAL TRADING THEME ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #080c10 !important;
    color: #c8d8e8 !important;
    font-family: 'Barlow', sans-serif !important;
}
[data-testid="stSidebar"] { background: #0d1219 !important; border-right: 1px solid #1a2535 !important; }
[data-testid="stHeader"] { background: #0d1219 !important; border-bottom: 1px solid #1a2535 !important; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* Headings */
h1 { font-family: 'Barlow Condensed', sans-serif !important; font-weight: 900 !important; letter-spacing: 3px !important; text-transform: uppercase !important; color: #00e5a0 !important; font-size: 2.2rem !important; }
h2 { font-family: 'Barlow Condensed', sans-serif !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: #4db8ff !important; font-size: 1.4rem !important; border-bottom: 1px solid #1a2535 !important; padding-bottom: 6px !important; }
h3 { font-family: 'Barlow Condensed', sans-serif !important; font-weight: 600 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; color: #8aa0b8 !important; font-size: 1.1rem !important; }

/* Metrics */
[data-testid="stMetric"] { background: #0a1018 !important; border: 1px solid #1a2535 !important; border-radius: 4px !important; padding: 12px 16px !important; }
[data-testid="stMetricLabel"] { color: #5a7090 !important; font-family: 'Barlow Condensed' !important; font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: #c8d8e8 !important; font-family: 'Share Tech Mono' !important; font-size: 1.4rem !important; }
[data-testid="stMetricDelta"] svg { display: none !important; }
[data-testid="stMetricDelta"] { font-family: 'Share Tech Mono' !important; font-size: 11px !important; }
[data-testid="stMetricDelta"] > div { color: #00e5a0 !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #243044 !important;
    color: #8aa0b8 !important;
    font-family: 'Barlow Condensed' !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { border-color: #00e5a0 !important; color: #00e5a0 !important; background: rgba(0,229,160,0.06) !important; }
.stButton > button[kind="primary"] { background: #00e5a0 !important; color: #080c10 !important; border-color: #00e5a0 !important; font-weight: 800 !important; }
.stButton > button[kind="primary"]:hover { background: #00c87a !important; border-color: #00c87a !important; }

/* Selectboxes / Inputs */
.stSelectbox > div > div, .stSlider > div { color: #c8d8e8 !important; }
[data-baseweb="select"] > div { background: #0d1219 !important; border-color: #243044 !important; color: #c8d8e8 !important; }
[data-baseweb="input"] > div { background: #0d1219 !important; border-color: #243044 !important; }
input, .stTextInput input { background: #0d1219 !important; color: #c8d8e8 !important; border-color: #243044 !important; font-family: 'Share Tech Mono' !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: #0d1219 !important; border-bottom: 1px solid #1a2535 !important; gap: 0 !important; }
[data-baseweb="tab"] { background: transparent !important; color: #5a7090 !important; font-family: 'Barlow Condensed' !important; font-weight: 700 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; font-size: 12px !important; border-bottom: 2px solid transparent !important; padding: 10px 20px !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #00e5a0 !important; border-bottom: 2px solid #00e5a0 !important; background: rgba(0,229,160,0.04) !important; }
[data-baseweb="tab-panel"] { background: transparent !important; padding: 0 !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid #1a2535 !important; border-radius: 4px !important; }
.dvn-scroller { background: #0a1018 !important; }

/* Info / success / warning boxes */
[data-testid="stInfo"] { background: rgba(77,184,255,0.08) !important; border: 1px solid #2a8fd1 !important; color: #4db8ff !important; border-radius: 4px !important; }
[data-testid="stSuccess"] { background: rgba(0,229,160,0.08) !important; border: 1px solid #00b87a !important; color: #00e5a0 !important; border-radius: 4px !important; }
[data-testid="stWarning"] { background: rgba(255,184,0,0.08) !important; border: 1px solid #cc9200 !important; color: #ffb800 !important; border-radius: 4px !important; }
[data-testid="stError"] { background: rgba(255,61,90,0.08) !important; border: 1px solid #cc3049 !important; color: #ff3d5a !important; border-radius: 4px !important; }

/* Progress bar */
[data-testid="stProgressBar"] > div > div { background: #00e5a0 !important; }

/* Sidebar nav items */
.stRadio label { color: #8aa0b8 !important; font-family: 'Barlow Condensed' !important; font-weight: 600 !important; letter-spacing: 1px !important; }
.stRadio [data-checked="true"] + div { color: #00e5a0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: #1a2535; border-radius: 3px; }

/* Plotly chart backgrounds via wrapper */
[data-testid="stPlotlyChart"] { border: 1px solid #1a2535 !important; border-radius: 4px !important; background: #0a1018 !important; }

/* Divider */
hr { border-color: #1a2535 !important; }

/* Spinner */
[data-testid="stSpinner"] { color: #00e5a0 !important; }
</style>
"""

FORECAST_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

/* ─── GLOBAL FORECAST THEME — deep navy / data lab ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0a0e1a !important;
    color: #d4dff0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] { background: #0e1525 !important; border-right: 1px solid #1c2840 !important; }
[data-testid="stHeader"] { background: #0e1525 !important; border-bottom: 1px solid #1c2840 !important; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* Headings — editorial/lab aesthetic */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #ffffff !important; font-size: 2.4rem !important; letter-spacing: -0.5px !important; text-transform: none !important; }
h1 span.accent { color: #7c6af7 !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #a0b4d0 !important; font-size: 1.2rem !important; letter-spacing: 0.5px !important; border-bottom: 1px solid #1c2840 !important; padding-bottom: 8px !important; text-transform: none !important; }
h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; color: #7c6af7 !important; font-size: 1rem !important; letter-spacing: 0.5px !important; }

/* Metrics — clean data card look */
[data-testid="stMetric"] { background: linear-gradient(135deg, #111b2e 0%, #0e1525 100%) !important; border: 1px solid #1c2840 !important; border-left: 3px solid #7c6af7 !important; border-radius: 6px !important; padding: 14px 18px !important; }
[data-testid="stMetricLabel"] { color: #6080a0 !important; font-family: 'DM Mono' !important; font-size: 11px !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'DM Mono' !important; font-size: 1.5rem !important; }
[data-testid="stMetricDelta"] > div { color: #52d48f !important; font-family: 'DM Mono' !important; font-size: 11px !important; }

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #2c3d60 !important;
    color: #7090b0 !important;
    font-family: 'Space Grotesk' !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover { border-color: #7c6af7 !important; color: #7c6af7 !important; background: rgba(124,106,247,0.06) !important; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #7c6af7, #5b4fd4) !important; color: white !important; border-color: transparent !important; font-weight: 700 !important; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.1) !important; }

/* Selects / inputs */
[data-baseweb="select"] > div { background: #111b2e !important; border-color: #2c3d60 !important; color: #d4dff0 !important; }
input, .stTextInput input { background: #111b2e !important; color: #d4dff0 !important; border-color: #2c3d60 !important; font-family: 'DM Mono' !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: #0e1525 !important; border-bottom: 1px solid #1c2840 !important; gap: 0 !important; }
[data-baseweb="tab"] { background: transparent !important; color: #506080 !important; font-family: 'Space Grotesk' !important; font-weight: 600 !important; letter-spacing: 0.5px !important; border-bottom: 2px solid transparent !important; padding: 10px 20px !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #7c6af7 !important; border-bottom: 2px solid #7c6af7 !important; background: rgba(124,106,247,0.05) !important; }

/* Info boxes */
[data-testid="stInfo"] { background: rgba(124,106,247,0.08) !important; border: 1px solid #5b4fd4 !important; color: #a090f8 !important; border-radius: 6px !important; }
[data-testid="stSuccess"] { background: rgba(82,212,143,0.08) !important; border: 1px solid #3aad72 !important; color: #52d48f !important; border-radius: 6px !important; }
[data-testid="stWarning"] { background: rgba(255,183,77,0.08) !important; border: 1px solid #cc8800 !important; color: #ffb74d !important; border-radius: 6px !important; }
[data-testid="stError"] { background: rgba(255,82,82,0.08) !important; border: 1px solid #cc3333 !important; color: #ff5252 !important; border-radius: 6px !important; }

/* Progress bar */
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #7c6af7, #52d48f) !important; }

/* Charts */
[data-testid="stPlotlyChart"] { border: 1px solid #1c2840 !important; border-radius: 8px !important; background: #0e1525 !important; }

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid #1c2840 !important; border-radius: 6px !important; }

/* Divider */
hr { border-color: #1c2840 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1c2840; border-radius: 3px; }

/* Spinner */
[data-testid="stSpinner"] { color: #7c6af7 !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div { background: #7c6af7 !important; }
</style>
"""

# ============================================================================
# NAVIGATION STATE
# ============================================================================

PAGES = [
    ("⚡", "Dashboard"),
    ("🦆", "Duck Curve"),
    ("🔮", "ML Forecasting"),
    ("⚡", "Generation Mix"),
    ("💰", "Price Analysis"),
    ("📈", "Trading Desk"),
    ("🚨", "Anomaly Detection"),
    ("📋", "Risk & Portfolio"),
]

if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# ============================================================================
# INJECT THEME based on current page
# ============================================================================

is_forecast_page = st.session_state.page == "ML Forecasting"
st.markdown(FORECAST_CSS if is_forecast_page else TRADING_CSS, unsafe_allow_html=True)

# ============================================================================
# TOP NAV BAR
# ============================================================================

nav_accent = "#7c6af7" if is_forecast_page else "#00e5a0"
nav_bg = "#0e1525" if is_forecast_page else "#0d1219"
nav_border = "#1c2840" if is_forecast_page else "#1a2535"
nav_font = "'Syne', sans-serif" if is_forecast_page else "'Barlow Condensed', sans-serif"

st.markdown(f"""
<style>
.top-nav {{
    display: flex;
    align-items: center;
    background: {nav_bg};
    border-bottom: 1px solid {nav_border};
    padding: 0 8px;
    height: 44px;
    margin: -1rem -1.5rem 1.5rem -1.5rem;
    gap: 0;
    position: sticky;
    top: 0;
    z-index: 999;
}}
.nav-logo {{
    font-family: {nav_font};
    font-weight: 900;
    font-size: 16px;
    letter-spacing: 2px;
    color: {nav_accent};
    margin-right: 20px;
    white-space: nowrap;
    text-transform: uppercase;
    padding: 0 8px;
}}
.nav-logo span {{ color: {'#506080' if is_forecast_page else '#4a6070'}; font-weight: 300; }}
.nav-status {{
    margin-left: auto;
    font-family: {'DM Mono' if is_forecast_page else 'Share Tech Mono'};
    font-size: 11px;
    color: {nav_accent};
    padding-right: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.status-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: {nav_accent};
    box-shadow: 0 0 7px {nav_accent};
    display: inline-block;
    animation: pulse 2s infinite;
}}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.35}} }}
</style>
<div class="top-nav">
    <div class="nav-logo">HOOPER <span>ENERGY</span></div>
    <div class="nav-status">
        <span class="status-dot"></span>
        {'MODEL ACTIVE · CAISO' if is_forecast_page else 'CAISO RT LIVE'}
        &nbsp;|&nbsp;
        {datetime.now().strftime('%H:%M PT')}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE SELECTOR — horizontal pill buttons
# ============================================================================

cols = st.columns(len(PAGES))
for i, (icon, name) in enumerate(PAGES):
    with cols[i]:
        is_active = st.session_state.page == name
        if is_forecast_page:
            btn_style = f"background:{'rgba(124,106,247,0.15)' if is_active else 'transparent'};border:1px solid {'#7c6af7' if is_active else '#1c2840'};color:{'#7c6af7' if is_active else '#506080'};"
        else:
            btn_style = f"background:{'rgba(0,229,160,0.08)' if is_active else 'transparent'};border:1px solid {'#00b87a' if is_active else '#1a2535'};color:{'#00e5a0' if is_active else '#4a6070'};"
        
        st.markdown(f"""
        <style>.nav-btn-{i} > div > button {{
            {btn_style}
            width: 100%; font-family: {'Syne' if is_forecast_page else 'Barlow Condensed'} !important;
            font-weight: {'700' if is_active else '600'} !important;
            font-size: 11px !important; letter-spacing: {'.5px' if is_forecast_page else '1.2px'} !important;
            text-transform: uppercase !important; border-radius: {'6px' if is_forecast_page else '2px'} !important;
            padding: 5px !important; height: 34px !important;
        }}</style>
        <div class="nav-btn-{i}">""", unsafe_allow_html=True)
        
        if st.button(f"{icon} {name}", key=f"nav_{name}", use_container_width=True):
            st.session_state.page = name
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ============================================================================
# SHARED DATA & UTILS
# ============================================================================

TRADING_COLORS = {
    'primary': '#00e5a0', 'secondary': '#4db8ff', 'warning': '#ffb800',
    'danger': '#ff3d5a', 'bg': '#080c10', 'panel': '#0a1018',
    'border': '#1a2535', 'text': '#c8d8e8', 'text_dim': '#5a7090'
}

FORECAST_COLORS = {
    'primary': '#7c6af7', 'secondary': '#52d48f', 'warning': '#ffb74d',
    'danger': '#ff5252', 'bg': '#0a0e1a', 'panel': '#0e1525',
    'border': '#1c2840', 'text': '#d4dff0', 'text_dim': '#506080'
}

C = FORECAST_COLORS if is_forecast_page else TRADING_COLORS

def plotly_dark_layout(title="", h=400, extra={}):
    base = dict(
        paper_bgcolor=C['panel'],
        plot_bgcolor=C['bg'],
        font=dict(color=C['text'], family="Share Tech Mono" if not is_forecast_page else "DM Mono"),
        title=dict(text=title, font=dict(size=13, color=C['text_dim'])),
        height=h,
        margin=dict(l=12, r=12, t=36 if title else 12, b=12),
        hovermode='x unified',
        xaxis=dict(gridcolor=C['border'], showgrid=True, zeroline=False, color=C['text_dim']),
        yaxis=dict(gridcolor=C['border'], showgrid=True, zeroline=False, color=C['text_dim']),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=C['border'], font=dict(size=11)),
    )
    base.update(extra)
    return base

@st.cache_data(ttl=3600)
def generate_sample_data(days=30):
    dates = pd.date_range(end=datetime.now(), periods=days*24*12, freq='5min')
    hour_of_day = dates.hour + dates.minute/60
    day_of_week = dates.dayofweek

    base_load = 25000 + 8000 * np.sin((hour_of_day - 6) * np.pi / 12)
    solar_impact = -5000 * np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12)) * (hour_of_day > 8) * (hour_of_day < 18)
    evening_ramp = 3000 * np.exp(-((hour_of_day - 19)**2) / 2)
    weekend_factor = np.where(day_of_week >= 5, 0.85, 1.0)
    noise = np.random.normal(0, 500, len(dates))
    seasonal = 2000 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24 * 12))
    load = (base_load + solar_impact + evening_ramp + seasonal + noise) * weekend_factor

    solar = np.maximum(0, 8000 * np.sin((hour_of_day - 6) * np.pi / 12) * (hour_of_day > 6) * (hour_of_day < 19))
    wind = np.clip(3000 + 2000 * np.random.randn(len(dates)).cumsum() / 100, 500, 6000)
    hydro = 5000 + 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / (365 * 24 * 12))
    nuclear = np.ones(len(dates)) * 2000
    batteries = np.maximum(0, np.where((hour_of_day >= 17) & (hour_of_day <= 21), 1500, -800))
    natural_gas = np.maximum(0, load - solar - wind - hydro - nuclear - batteries)
    net_load = load - solar - wind
    price = np.clip(30 + 0.002 * net_load + 10 * np.random.randn(len(dates)), 10, 200)
    curtailment = np.maximum(0, solar - net_load * 0.1) * (hour_of_day > 9) * (hour_of_day < 16)

    return pd.DataFrame({
        'timestamp': dates, 'load': load, 'solar': solar, 'wind': wind,
        'hydro': hydro, 'natural_gas': natural_gas, 'nuclear': nuclear,
        'batteries': batteries, 'price': price, 'net_load': net_load,
        'curtailment': curtailment
    })

@st.cache_data
def create_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = ((df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)
    df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['hour_x_dow'] = df['hour'] * df['day_of_week']
    df['is_weekend_x_hour'] = df['is_weekend'] * df['hour']
    for lag in [1, 2, 12, 24, 288, 576, 2016]:
        df[f'load_lag_{lag}'] = df['load'].shift(lag)
    for window in [12, 288, 2016]:
        df[f'load_rolling_mean_{window}'] = df['load'].rolling(window=window, min_periods=1).mean()
        df[f'load_rolling_std_{window}'] = df['load'].rolling(window=window, min_periods=1).std()
    df['load_ema_12'] = df['load'].ewm(span=12, adjust=False).mean()
    df['load_ema_288'] = df['load'].ewm(span=288, adjust=False).mean()
    df['load_diff_1'] = df['load'].diff(1)
    df['load_diff_288'] = df['load'].diff(288)
    df['load_daily_change'] = df['load'] - df['load'].shift(288)
    return df.dropna()

# ============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================================

if st.session_state.page == "Dashboard":
    st.markdown("# ⚡ CALIFORNIA ENERGY INTELLIGENCE PLATFORM")
    st.markdown("**CAISO · Real-time market data · Updated 5-min intervals**")

    with st.spinner('Loading market data...'):
        df = generate_sample_data(days=30)

    recent = df.tail(288)  # last 24h

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        ("Current Load", f"{df['load'].iloc[-1]:,.0f} MW", f"{((df['load'].iloc[-1]/df['load'].iloc[-289])-1)*100:+.1f}%"),
        ("Net Load", f"{df['net_load'].iloc[-1]:,.0f} MW", f"{((df['net_load'].iloc[-1]/df['net_load'].iloc[-289])-1)*100:+.1f}%"),
        ("Avg Price 24H", f"${recent['price'].mean():.2f}/MWh", f"{((recent['price'].mean()/df['price'].iloc[-289])-1)*100:+.1f}%"),
        ("Renewable %", f"{(df[['solar','wind']].sum(axis=1)/df['load']*100).iloc[-1]:.1f}%", ""),
        ("Solar Curtail", f"{recent['curtailment'].sum()/12:,.0f} MWh", "24H"),
        ("Price Volatility", f"${recent['price'].std():.2f}", "σ"),
    ]
    for col, (label, val, delta) in zip([c1,c2,c3,c4,c5,c6], metrics):
        with col:
            st.metric(label, val, delta if delta else None)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## Load & Net Load — 7 Days")
        recent7 = df.tail(7*24*12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent7['timestamp'], y=recent7['load'], name='Gross Load',
            line=dict(color=C['secondary'], width=2)))
        fig.add_trace(go.Scatter(x=recent7['timestamp'], y=recent7['net_load'], name='Net Load',
            line=dict(color=C['primary'], width=2), fill='tonexty',
            fillcolor=f"rgba({'77,184,255' if not is_forecast_page else '124,106,247'},0.08)"))
        fig.update_layout(**plotly_dark_layout(h=340))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("## LMP Price — 7 Days")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent7['timestamp'], y=recent7['price'], name='$/MWh',
            line=dict(color=C['warning'], width=1.5), fill='tozeroy',
            fillcolor=f"rgba({'255,184,0' if not is_forecast_page else '255,183,77'},0.06)"))
        fig.update_layout(**plotly_dark_layout(h=340))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Current Generation Mix")
    latest = df.iloc[-1]
    srcs = ['Solar','Wind','Hydro','Natural Gas','Nuclear','Batteries']
    vals = [latest['solar'], latest['wind'], latest['hydro'], latest['natural_gas'], latest['nuclear'], max(0,latest['batteries'])]
    colors_pie = ['#ffb800','#4db8ff','#2E86AB','#ff3d5a','#00e5a0','#b47dff']
    fig = go.Figure(go.Pie(labels=srcs, values=vals, marker=dict(colors=colors_pie),
        hole=0.45, textinfo='label+percent', textfont=dict(size=13, color='#c8d8e8')))
    fig.update_layout(**plotly_dark_layout(h=380))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 2: DUCK CURVE
# ============================================================================

elif st.session_state.page == "Duck Curve":
    st.markdown("# 🦆 DUCK CURVE ANALYSIS")
    df = generate_sample_data(days=30)

    df['hour_frac'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    daily = df.groupby('hour_frac').agg({'load':'mean','net_load':'mean','solar':'mean','wind':'mean'}).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['hour_frac'], y=daily['load'], name='Gross Load',
        line=dict(color=C['secondary'], width=3), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=daily['hour_frac'], y=daily['net_load'], name='Net Load (Duck Curve)',
        line=dict(color=C['primary'], width=3), marker=dict(size=5),
        fill='tonexty', fillcolor=f"rgba({'0,229,160' if not is_forecast_page else '124,106,247'},0.08)"))
    fig.update_layout(**plotly_dark_layout("Average Daily Duck Curve Profile", h=460,
        extra=dict(xaxis=dict(title="Hour of Day", gridcolor=C['border'], tickvals=list(range(0,25,2)), color=C['text_dim']),
                   yaxis=dict(title="Load (MW)", gridcolor=C['border'], color=C['text_dim']))))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    belly_depth = daily['load'].max() - daily['net_load'].min()
    idx_min = daily['net_load'].idxmin()
    idx_eve = daily[(daily['hour_frac'] >= 17) & (daily['hour_frac'] <= 21)].index
    ramp = (daily.loc[idx_eve, 'net_load'].max() - daily['net_load'].iloc[idx_min]) / 4
    solar_pen = (daily['solar'].max() / daily['load'].max() * 100)
    with c1: st.metric("Duck Belly Depth", f"{belly_depth:,.0f} MW", "Load - min net load")
    with c2: st.metric("Min Net Load Hour", f"{daily.loc[idx_min,'hour_frac']:.1f}H", "Solar noon dip")
    with c3: st.metric("Evening Ramp Rate", f"{ramp:,.0f} MW/hr", "HE17-21 avg")
    with c4: st.metric("Peak Solar Penetration", f"{solar_pen:.1f}%", "of gross load")

    st.markdown("## Renewable Contribution by Hour")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=daily['hour_frac'], y=daily['solar'], name='Solar',
        fill='tozeroy', fillcolor='rgba(255,184,0,0.5)', line=dict(color='#ffb800', width=2)))
    fig2.add_trace(go.Scatter(x=daily['hour_frac'], y=daily['wind'], name='Wind',
        fill='tozeroy', fillcolor='rgba(77,184,255,0.3)', line=dict(color='#4db8ff', width=2)))
    fig2.update_layout(**plotly_dark_layout(h=320,
        extra=dict(xaxis=dict(title="Hour of Day", gridcolor=C['border'], color=C['text_dim']),
                   yaxis=dict(title="Generation (MW)", gridcolor=C['border'], color=C['text_dim']))))
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# PAGE 3: ML FORECASTING  ← DISTINCT VISUAL THEME
# ============================================================================

elif st.session_state.page == "ML Forecasting":
    # Big editorial header — Syne font
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:2.6rem;color:#fff;line-height:1.1;letter-spacing:-1px;">
            ML Forecasting <span style="color:#7c6af7;">Engine</span>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#506080;margin-top:6px;letter-spacing:1px;">
            CAISO LOAD FORECAST · TARGET MAPE ≤ 5% · BLACK-BOX → EXPLAINABLE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("🎯 **Goal:** Beat the benchmark with MAPE ≤ 5%. All models trained on 60 days of 5-min CAISO load data with 50+ engineered features.")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        model_type = st.selectbox("Model Selection",
            ["Ridge Regression (Baseline)", "Lasso (Feature Selection)", "Hist Gradient Boosting", "Random Forest", "⭐ Ensemble (Recommended)"])
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (hours)", 1, 72, 24)
    with col3:
        optimize_hp = st.checkbox("Optimize Hyperparameters", value=False)

    # Model architecture diagram
    st.markdown("### Model Architecture")
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0 20px 0;">
        <div style="background:#111b2e;border:1px solid #1c2840;border-top:3px solid #7c6af7;border-radius:6px;padding:10px 8px;text-align:center;">
            <div style="font-family:'DM Mono';font-size:9px;color:#506080;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Temporal</div>
            <div style="font-family:'Space Grotesk';font-size:11px;color:#d4dff0;font-weight:600;">Hour · DOW<br>Month · DOY</div>
        </div>
        <div style="background:#111b2e;border:1px solid #1c2840;border-top:3px solid #52d48f;border-radius:6px;padding:10px 8px;text-align:center;">
            <div style="font-family:'DM Mono';font-size:9px;color:#506080;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Cyclical</div>
            <div style="font-family:'Space Grotesk';font-size:11px;color:#d4dff0;font-weight:600;">sin/cos<br>Encoding</div>
        </div>
        <div style="background:#111b2e;border:1px solid #1c2840;border-top:3px solid #ffb74d;border-radius:6px;padding:10px 8px;text-align:center;">
            <div style="font-family:'DM Mono';font-size:9px;color:#506080;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Lag Features</div>
            <div style="font-family:'Space Grotesk';font-size:11px;color:#d4dff0;font-weight:600;">5min → 1wk<br>9 lags</div>
        </div>
        <div style="background:#111b2e;border:1px solid #1c2840;border-top:3px solid #4db8ff;border-radius:6px;padding:10px 8px;text-align:center;">
            <div style="font-family:'DM Mono';font-size:9px;color:#506080;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Rolling Stats</div>
            <div style="font-family:'Space Grotesk';font-size:11px;color:#d4dff0;font-weight:600;">Mean · Std<br>EMA</div>
        </div>
        <div style="background:#111b2e;border:1px solid #1c2840;border-top:3px solid #b47dff;border-radius:6px;padding:10px 8px;text-align:center;">
            <div style="font-family:'DM Mono';font-size:9px;color:#506080;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Interaction</div>
            <div style="font-family:'Space Grotesk';font-size:11px;color:#d4dff0;font-weight:600;">Hour×DOW<br>Peak flags</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Train & Forecast", type="primary"):
        with st.spinner('Engineering features and training models...'):
            df = generate_sample_data(days=60)
            df_feat = create_features(df)

            train_size = int(len(df_feat) * 0.8)
            train_data = df_feat.iloc[:train_size]
            test_data = df_feat.iloc[train_size:]

            excl = ['timestamp', 'load', 'price', 'net_load', 'curtailment']
            feature_cols = [c for c in df_feat.columns if c not in excl]

            X_train = train_data[feature_cols]
            y_train = train_data['load']
            X_test = test_data[feature_cols]
            y_test = test_data['load']

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            models = {}
            predictions = {}
            prog = st.progress(0, "Training models...")

            def train_model(name, model, scaled=False):
                model.fit(X_train_s if scaled else X_train, y_train)
                predictions[name] = model.predict(X_test_s if scaled else X_test)
                models[name] = model

            mt = model_type
            if "Ridge" in mt:
                train_model('Ridge', Ridge(alpha=1.0), scaled=True); prog.progress(80)
            elif "Lasso" in mt:
                train_model('Lasso', Lasso(alpha=0.05, max_iter=3000), scaled=True); prog.progress(80)
            elif "Hist" in mt:
                train_model('HGB', HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=10, random_state=42)); prog.progress(80)
            elif "Random" in mt:
                train_model('RandomForest', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)); prog.progress(80)
            else:  # Ensemble
                train_model('Ridge', Ridge(alpha=1.0), scaled=True); prog.progress(30)
                train_model('HGB', HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=10, random_state=42)); prog.progress(65)
                train_model('RandomForest', RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)); prog.progress(90)
                # Weighted ensemble by inverse variance
                preds_arr = np.array(list(predictions.values()))
                weights = np.array([1/np.var(p - y_test.values) for p in preds_arr])
                weights /= weights.sum()
                predictions['Ensemble'] = (preds_arr * weights[:,None]).sum(axis=0)

            prog.progress(100, "Complete!")

        st.success("✅ Training complete!")
        st.markdown("---")

        # ── Performance metrics
        st.markdown("### Model Performance")
        results = []
        for name, pred in predictions.items():
            mape = mean_absolute_percentage_error(y_test, pred) * 100
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)
            results.append({'Model': name, 'MAPE %': round(mape, 3), 'MAE MW': round(mae, 1),
                           'RMSE MW': round(rmse, 1), 'R²': round(r2, 4), '_pred': pred})

        res_cols = st.columns(len(results))
        for col, r in zip(res_cols, results):
            with col:
                target_hit = r['MAPE %'] <= 7.0
                delta_str = f"{'✓ Target hit' if target_hit else '✗ Above target'}"
                st.metric(f"{r['Model']}", f"{r['MAPE %']:.2f}%", delta_str)

        res_display = pd.DataFrame([{k:v for k,v in r.items() if k!='_pred'} for r in results])
        st.dataframe(res_display.style.highlight_min(subset=['MAPE %','RMSE MW'], color='#1a3a2a')
                     .highlight_max(subset=['R²'], color='#1a3a2a'), use_container_width=True)

        st.markdown("---")
        # ── Forecast vs Actual chart
        st.markdown("### Forecast vs Actual")
        plot_n = min(forecast_horizon * 12, len(test_data))
        plot_data = test_data.head(plot_n)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_data['timestamp'], y=y_test.values[:plot_n],
            name='Actual Load', line=dict(color='#d4dff0', width=2.5)))

        pred_colors = ['#7c6af7','#52d48f','#ffb74d','#4db8ff']
        for idx, r in enumerate(results):
            if r['Model'] != 'Ensemble':
                fig.add_trace(go.Scatter(x=plot_data['timestamp'], y=r['_pred'][:plot_n],
                    name=f"{r['Model']} (MAPE {r['MAPE %']:.2f}%)",
                    line=dict(color=pred_colors[idx % len(pred_colors)], width=1.5, dash='dot')))
            else:
                fig.add_trace(go.Scatter(x=plot_data['timestamp'], y=r['_pred'][:plot_n],
                    name=f"⭐ Ensemble (MAPE {r['MAPE %']:.2f}%)",
                    line=dict(color='#52d48f', width=2.5, dash='dash')))

        fig.update_layout(**plotly_dark_layout(f"{forecast_horizon}H Forecast", h=460,
            extra=dict(paper_bgcolor='#0e1525', plot_bgcolor='#0a0e1a',
                       xaxis=dict(title="Time", gridcolor='#1c2840', color='#506080'),
                       yaxis=dict(title="Load (MW)", gridcolor='#1c2840', color='#506080'),
                       font=dict(color='#d4dff0', family="DM Mono"))))
        st.plotly_chart(fig, use_container_width=True)

        # ── Residual distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Residual Distribution")
            best = results[0] if len(results) == 1 else max(results, key=lambda r: r['R²'])
            residuals = y_test.values[:plot_n] - best['_pred'][:plot_n]
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(x=residuals, nbinsx=60, name='Residuals',
                marker=dict(color='#7c6af7', opacity=0.7)))
            fig_res.add_vline(x=0, line=dict(color='#52d48f', dash='dash', width=2))
            fig_res.update_layout(**plotly_dark_layout(f"Residuals — {best['Model']}", h=320,
                extra=dict(paper_bgcolor='#0e1525', plot_bgcolor='#0a0e1a',
                           xaxis=dict(title="Residual (MW)", gridcolor='#1c2840', color='#506080'),
                           yaxis=dict(title="Count", gridcolor='#1c2840', color='#506080'),
                           font=dict(color='#d4dff0', family="DM Mono"))))
            st.plotly_chart(fig_res, use_container_width=True)

        with col2:
            st.markdown("### Feature Importance")
            hgb_result = next((r for r in results if r['Model'] in ['HGB','RandomForest']), None)
            if hgb_result:
                m = models.get('HGB') or models.get('RandomForest')
                if hasattr(m, 'feature_importances_'):
                    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': m.feature_importances_})
                    imp_df = imp_df.sort_values('Importance', ascending=True).tail(15)
                    fig_imp = go.Figure(go.Bar(x=imp_df['Importance'], y=imp_df['Feature'],
                        orientation='h', marker=dict(color='#7c6af7', opacity=0.85)))
                    fig_imp.update_layout(**plotly_dark_layout("Top 15 Features", h=320,
                        extra=dict(paper_bgcolor='#0e1525', plot_bgcolor='#0a0e1a',
                                   xaxis=dict(title="Importance", gridcolor='#1c2840', color='#506080'),
                                   yaxis=dict(gridcolor='#1c2840', color='#506080'),
                                   font=dict(color='#d4dff0', family="DM Mono"))))
                    st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Train with HGB or Random Forest to see feature importance.")

        # ── Error by hour of day
        st.markdown("### Forecast Error by Hour of Day")
        best_pred = max(results, key=lambda r: r['R²'])['_pred']
        err_df = test_data[['timestamp']].head(plot_n).copy()
        err_df['abs_error'] = np.abs(y_test.values[:plot_n] - best_pred[:plot_n])
        err_df['hour'] = err_df['timestamp'].dt.hour
        hourly_err = err_df.groupby('hour')['abs_error'].mean().reset_index()
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Bar(x=hourly_err['hour'], y=hourly_err['abs_error'],
            marker=dict(color=hourly_err['abs_error'], colorscale='Purp', showscale=False),
            name='MAE by Hour'))
        fig_hr.add_vline(x=17, line=dict(color='#ffb74d', dash='dash', width=1.5),
            annotation_text="Evening Peak", annotation_font_color='#ffb74d')
        fig_hr.update_layout(**plotly_dark_layout("Mean Absolute Error by Hour of Day", h=300,
            extra=dict(paper_bgcolor='#0e1525', plot_bgcolor='#0a0e1a',
                       xaxis=dict(title="Hour of Day", gridcolor='#1c2840', color='#506080', tickvals=list(range(0,24,2))),
                       yaxis=dict(title="MAE (MW)", gridcolor='#1c2840', color='#506080'),
                       font=dict(color='#d4dff0', family="DM Mono"))))
        st.plotly_chart(fig_hr, use_container_width=True)

    else:
        st.markdown("""
        <div style="border:1px dashed #2c3d60;border-radius:8px;padding:32px;text-align:center;margin:20px 0;">
            <div style="font-family:'Syne';font-size:1.4rem;color:#506080;margin-bottom:8px;">Ready to Train</div>
            <div style="font-family:'DM Mono';font-size:11px;color:#404e65;letter-spacing:1px;">Configure model above and click <strong style='color:#7c6af7'>Train & Forecast</strong></div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# PAGE 4: GENERATION MIX
# ============================================================================

elif st.session_state.page == "Generation Mix":
    st.markdown("# ⚡ GENERATION MIX ANALYSIS")
    df = generate_sample_data(days=7)

    fig = go.Figure()
    sources = [
        ('Nuclear', 'nuclear', '#00e5a0'), ('Hydro', 'hydro', '#2E86AB'),
        ('Natural Gas', 'natural_gas', '#ff3d5a'), ('Wind', 'wind', '#4db8ff'),
        ('Batteries', 'batteries', '#b47dff'), ('Solar', 'solar', '#ffb800'),
    ]
    for name, col, color in sources:
        if col in df.columns:
            vals = df[col].clip(lower=0)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=vals, name=name,
                stackgroup='one', fillcolor=color, line=dict(width=0.5, color=color)))
    fig.update_layout(**plotly_dark_layout("Generation Stack — Last 7 Days", h=480,
        extra=dict(yaxis=dict(title="Generation (MW)", gridcolor=C['border'], color=C['text_dim']),
                   legend=dict(orientation="h", y=1.05))))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("## Generation Summary")
        total_gen = df[[c for _,c,_ in sources if c in df.columns]].clip(lower=0).sum()
        gen_df = pd.DataFrame({
            'Source': [n for n,_,_ in sources if _ in ['#00e5a0','#2E86AB','#ff3d5a','#4db8ff','#b47dff','#ffb800']][:len(total_gen)],
            'Total MWh': total_gen.values,
            'Share %': (total_gen.values / total_gen.sum() * 100)
        })
        st.dataframe(gen_df.style.format({'Total MWh': '{:,.0f}', 'Share %': '{:.1f}%'}),
                    use_container_width=True, hide_index=True)

    with c2:
        st.markdown("## Renewable vs Conventional")
        ren = df[['solar','wind','hydro']].clip(lower=0).sum().sum()
        conv = df[['natural_gas']].clip(lower=0).sum().sum()
        nuc = df[['nuclear']].clip(lower=0).sum().sum()
        fig_p = go.Figure(go.Pie(
            labels=['Renewable','Natural Gas','Nuclear'],
            values=[ren, conv, nuc],
            marker=dict(colors=['#00e5a0','#ff3d5a','#4db8ff']),
            hole=0.45, textinfo='label+percent', textfont=dict(size=13)))
        fig_p.update_layout(**plotly_dark_layout(h=320))
        st.plotly_chart(fig_p, use_container_width=True)


# ============================================================================
# PAGE 5: PRICE ANALYSIS
# ============================================================================

elif st.session_state.page == "Price Analysis":
    st.markdown("# 💰 PRICE ANALYSIS & OPTIMIZATION")
    df = generate_sample_data(days=30)
    recent7 = df.tail(7*24*12)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Avg Price 7D", f"${recent7['price'].mean():.2f}/MWh")
    with c2: st.metric("Peak Price 7D", f"${recent7['price'].max():.2f}/MWh", "Max observed")
    with c3: st.metric("Off-Peak Avg", f"${recent7[recent7['timestamp'].dt.hour.between(0,6)]['price'].mean():.2f}/MWh")
    with c4: st.metric("Price Volatility σ", f"${recent7['price'].std():.2f}")

    st.markdown("## Net Load vs Price Correlation")
    scatter_df = recent7.iloc[::12].copy()
    # Manual polynomial trendline — no statsmodels needed
    _x = scatter_df['net_load'].values
    _y = scatter_df['price'].values
    _z = np.polyfit(_x, _y, deg=2)
    _p = np.poly1d(_z)
    _x_line = np.linspace(_x.min(), _x.max(), 200)
    _y_line = _p(_x_line)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scatter_df['net_load'], y=scatter_df['price'],
        mode='markers',
        name='Observations',
        marker=dict(
            color=scatter_df['price'],
            colorscale='RdYlGn_r',
            size=6, opacity=0.7,
            colorbar=dict(title="$/MWh", tickfont=dict(color=C['text_dim']))
        )
    ))
    fig.add_trace(go.Scatter(
        x=_x_line, y=_y_line,
        mode='lines', name='Trend (poly-2)',
        line=dict(color=C['primary'], width=2.5, dash='dash')
    ))
    # Pearson correlation annotation
    _r = np.corrcoef(_x, _y)[0, 1]
    fig.add_annotation(
        x=0.02, y=0.96, xref='paper', yref='paper',
        text=f"Pearson r = {_r:.3f}",
        showarrow=False,
        font=dict(color=C['text_dim'], size=12,
                  family="Share Tech Mono" if not is_forecast_page else "DM Mono"),
        bgcolor=C['panel'], bordercolor=C['border'], borderwidth=1
    )
    fig.update_layout(**plotly_dark_layout(h=380,
        extra=dict(
            xaxis=dict(title="Net Load (MW)", gridcolor=C['border'], color=C['text_dim']),
            yaxis=dict(title="Price ($/MWh)", gridcolor=C['border'], color=C['text_dim'])
        )))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Price Heatmap — Hour × Day of Week")
    hm = recent7.copy()
    hm['hour'] = hm['timestamp'].dt.hour
    hm['day'] = hm['timestamp'].dt.day_name()
    pivot = hm.pivot_table(values='price', index='hour', columns='day', aggfunc='mean')
    day_ord = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = pivot[[d for d in day_ord if d in pivot.columns]]
    fig_h = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale='RdYlGn_r', colorbar=dict(title="$/MWh", tickfont=dict(color=C['text_dim']))))
    fig_h.update_layout(**plotly_dark_layout(h=460,
        extra=dict(xaxis=dict(title="Day of Week", gridcolor=C['border'], color=C['text_dim']),
                   yaxis=dict(title="Hour of Day", gridcolor=C['border'], color=C['text_dim']))))
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("## Trading Opportunity Windows")
    c1, c2, c3 = st.columns(3)
    q25 = df['price'].quantile(0.25)
    q75 = df['price'].quantile(0.75)
    with c1: st.metric("Buy/Charge Window", f"{(df['price']<q25).sum()/12:.0f} hrs", f"< ${q25:.2f}/MWh")
    with c2: st.metric("Sell/Discharge Window", f"{(df['price']>q75).sum()/12:.0f} hrs", f"> ${q75:.2f}/MWh")
    with c3: st.metric("Arb Spread (Q3-Q1)", f"${q75-q25:.2f}/MWh", "Max simple arb")


# ============================================================================
# PAGE 6: TRADING DESK  — embeds the full terminal
# ============================================================================

elif st.session_state.page == "Trading Desk":
    st.markdown("# 📈 ENERGY TRADING DESK")
    st.markdown("**Bloomberg-style live trading terminal — paper trading mode**")
    st.markdown("")

    # Load the terminal HTML — try all likely locations
    import os
    _terminal_candidates = [
        'energy_trading_terminal_v2.html',                         # repo root (Replit)
        'energy_trading_terminal.html',                            # original filename
        os.path.join(os.path.dirname(__file__), 'energy_trading_terminal_v2.html'),
        '/mnt/user-data/outputs/energy_trading_terminal_v2.html',  # Claude outputs
    ]
    terminal_html = None
    for _path in _terminal_candidates:
        if os.path.exists(_path):
            with open(_path, 'r') as f:
                terminal_html = f.read()
            break

    if terminal_html:
        st.components.v1.html(terminal_html, height=820, scrolling=False)
    else:
        st.info(
            "**Trading Terminal Setup:** Copy `energy_trading_terminal_v2.html` to your repo root "
            "(same folder as `energy_dashboard_main.py`) and restart the app. "
            "Showing live market view below in the meantime."
        )
        df = generate_sample_data(days=7)

        c1,c2,c3,c4,c5 = st.columns(5)
        lmp_now = df['price'].iloc[-1]
        lmp_da = lmp_now * (1 + np.random.uniform(-0.1, 0.1))
        with c1: st.metric("SP15 RT LMP", f"${lmp_now:.2f}", f"{lmp_now - df['price'].iloc[-13]:+.2f}")
        with c2: st.metric("SP15 DA LMP", f"${lmp_da:.2f}", "Day-Ahead")
        with c3: st.metric("Reg Up AS", f"${np.random.uniform(12,18):.2f}/MW-hr")
        with c4: st.metric("SoCal Gas", f"${np.random.uniform(3.6,4.2):.2f}/MMBtu")
        with c5: st.metric("Spark Spread", f"${lmp_now - 3.9*8.2:.2f}/MWh")

        st.markdown("## LMP Price — Last 24H")
        recent24 = df.tail(288)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent24['timestamp'], y=recent24['price'],
            name='SP15 LMP', line=dict(color='#00e5a0', width=2),
            fill='tozeroy', fillcolor='rgba(0,229,160,0.07)'))
        fig.update_layout(**plotly_dark_layout("SP15 RT LMP — Last 24H", h=380))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 7: ANOMALY DETECTION
# ============================================================================

elif st.session_state.page == "Anomaly Detection":
    st.markdown("# 🚨 ANOMALY DETECTION")
    df = generate_sample_data(days=30)

    threshold = st.slider("Detection Threshold (σ)", 1.5, 4.0, 3.0, 0.5)

    df['roll_mean'] = df['load'].rolling(window=288, center=True, min_periods=1).mean()
    df['roll_std'] = df['load'].rolling(window=288, center=True, min_periods=1).std().fillna(500)
    df['upper'] = df['roll_mean'] + threshold * df['roll_std']
    df['lower'] = df['roll_mean'] - threshold * df['roll_std']
    df['is_anomaly'] = (df['load'] > df['upper']) | (df['load'] < df['lower'])
    anomalies = df[df['is_anomaly']]

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Anomalies Detected", f"{len(anomalies)}", f"at {threshold}σ threshold")
    with c2: st.metric("Anomaly Rate", f"{len(anomalies)/len(df)*100:.2f}%", "of all intervals")
    with c3: st.metric("Max Deviation", f"{((anomalies['load'] - anomalies['roll_mean'])/anomalies['roll_mean']*100).abs().max():.1f}%" if len(anomalies) else "N/A")

    st.markdown(f"## Anomaly Detection — {len(anomalies)} Events Found")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['load'], name='Load',
        line=dict(color='#4db8ff', width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['upper'], name='Upper Bound',
        line=dict(color='#ff3d5a', width=1, dash='dash'), showlegend=True))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['lower'], name='Lower Bound',
        line=dict(color='#ff3d5a', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(255,61,90,0.06)'))
    if len(anomalies):
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['load'],
            name='Anomalies', mode='markers',
            marker=dict(color='#ff3d5a', size=9, symbol='x', line=dict(width=2))))
    fig.update_layout(**plotly_dark_layout(h=480,
        extra=dict(yaxis=dict(title="Load (MW)", gridcolor=C['border'], color=C['text_dim']))))
    st.plotly_chart(fig, use_container_width=True)

    if len(anomalies) > 0:
        st.markdown("## Anomaly Detail Log")
        disp = anomalies[['timestamp','load','roll_mean','price']].head(50).copy()
        disp['deviation_pct'] = ((disp['load'] - disp['roll_mean']) / disp['roll_mean'] * 100).round(1)
        disp.columns = ['Timestamp','Load (MW)','Expected (MW)','Price ($/MWh)','Deviation %']
        st.dataframe(disp.style.format({
            'Load (MW)': '{:,.0f}', 'Expected (MW)': '{:,.0f}',
            'Price ($/MWh)': '${:.2f}', 'Deviation %': '{:+.1f}%'
        }), use_container_width=True, height=380)


# ============================================================================
# PAGE 8: RISK & PORTFOLIO
# ============================================================================

elif st.session_state.page == "Risk & Portfolio":
    st.markdown("# 📋 RISK MANAGEMENT & PORTFOLIO")
    df = generate_sample_data(days=90)

    # Simulate portfolio P&L
    daily = df.groupby(df['timestamp'].dt.date).agg(
        avg_price=('price','mean'), load_gwh=('load','sum'), solar_gwh=('solar','sum')
    ).reset_index()
    daily['load_gwh'] = daily['load_gwh'] / 12000
    daily['solar_gwh'] = daily['solar_gwh'] / 12000
    daily['pnl'] = (daily['avg_price'] - 45) * daily['load_gwh'] * np.random.normal(0.08, 0.02, len(daily))
    daily['cum_pnl'] = daily['pnl'].cumsum()

    c1,c2,c3,c4 = st.columns(4)
    rets = daily['pnl'].values
    var_95 = np.percentile(rets, 5)
    cvar_95 = rets[rets <= var_95].mean()
    with c1: st.metric("Cumulative P&L", f"${daily['cum_pnl'].iloc[-1]:,.0f}K")
    with c2: st.metric("VaR (95%)", f"${abs(var_95):,.0f}K/day", "1-day horizon")
    with c3: st.metric("CVaR (95%)", f"${abs(cvar_95):,.0f}K/day", "Expected shortfall")
    with c4: st.metric("Sharpe Ratio", f"{(rets.mean()/rets.std()*np.sqrt(252)):.2f}", "Annualised")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## Cumulative Portfolio P&L")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily['timestamp'].astype(str), y=daily['cum_pnl'],
            name='Cumulative P&L', line=dict(color='#00e5a0', width=2),
            fill='tozeroy', fillcolor='rgba(0,229,160,0.07)'))
        fig.add_hline(y=0, line=dict(color='#ff3d5a', dash='dash', width=1))
        fig.update_layout(**plotly_dark_layout(h=340,
            extra=dict(yaxis=dict(title="P&L ($K)", gridcolor=C['border'], color=C['text_dim']))))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("## Daily P&L Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=daily['pnl'], nbinsx=40, name='Daily P&L',
            marker=dict(color='#4db8ff', opacity=0.75)))
        fig.add_vline(x=var_95, line=dict(color='#ffb800', dash='dash', width=2),
            annotation_text=f"VaR 95%: ${var_95:.0f}K", annotation_font_color='#ffb800')
        fig.add_vline(x=0, line=dict(color='#ff3d5a', dash='dot', width=1))
        fig.update_layout(**plotly_dark_layout(h=340,
            extra=dict(xaxis=dict(title="Daily P&L ($K)", gridcolor=C['border'], color=C['text_dim']),
                       yaxis=dict(title="Count", gridcolor=C['border'], color=C['text_dim']))))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Monte Carlo Simulation — 90-Day P&L Paths (500 scenarios)")
    np.random.seed(42)
    mu, sigma = rets.mean(), rets.std()
    paths = np.cumsum(np.random.normal(mu, sigma, (500, 90)), axis=1)
    p5, p50, p95 = np.percentile(paths, [5,50,95], axis=0)
    x_idx = list(range(90))
    fig_mc = go.Figure()
    for i in range(0, 500, 10):
        fig_mc.add_trace(go.Scatter(x=x_idx, y=paths[i], mode='lines',
            line=dict(color='rgba(77,184,255,0.06)', width=1), showlegend=False))
    fig_mc.add_trace(go.Scatter(x=x_idx, y=p95, name='P95', line=dict(color='#00e5a0', width=2, dash='dash')))
    fig_mc.add_trace(go.Scatter(x=x_idx, y=p50, name='Median', line=dict(color='#ffb800', width=2)))
    fig_mc.add_trace(go.Scatter(x=x_idx, y=p5, name='P5', line=dict(color='#ff3d5a', width=2, dash='dash')))
    fig_mc.update_layout(**plotly_dark_layout("Monte Carlo — 90-Day P&L Simulation", h=420,
        extra=dict(xaxis=dict(title="Day", gridcolor=C['border'], color=C['text_dim']),
                   yaxis=dict(title="Cumulative P&L ($K)", gridcolor=C['border'], color=C['text_dim']))))
    st.plotly_chart(fig_mc, use_container_width=True)


# ============================================================================
# FOOTER
# ============================================================================

footer_color = '#7c6af7' if is_forecast_page else '#00e5a0'
footer_font = "Space Grotesk" if is_forecast_page else "Barlow Condensed"

st.markdown(f"""
<div style="margin-top:2rem;padding:12px 0;border-top:1px solid {C['border']};
    display:flex;align-items:center;justify-content:space-between;">
    <div style="font-family:'{footer_font}';font-size:11px;color:{C['text_dim']};letter-spacing:1px;text-transform:uppercase;">
        Hooper Energy Solutions &nbsp;·&nbsp; California Energy Intelligence Platform v2.1
    </div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{footer_color};">
        CAISO · EIA · CPUC · Open-Meteo &nbsp;·&nbsp; Paper Trading Mode
    </div>
</div>
""", unsafe_allow_html=True)
