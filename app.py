import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import pytz
import base64
import requests
import numpy as np
from datetime import datetime
from etl.fetcher import DataFetcher

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Trade AI: War Room", layout="wide")

WATCHLIST = [
    {"Ticker": "CBRE",   "Market": "US"},
    {"Ticker": "SAN",    "Market": "US"}, 
    {"Ticker": "CSCO",   "Market": "US"},
    {"Ticker": "RTO.L",  "Market": "UK"},
    {"Ticker": "SAN.MC", "Market": "ES"},
]

TOOLTIPS = {
    "RSI": "Relative Strength Index (0-100). Below 30 is 'Oversold', above 70 is 'Overbought'.",
    "AI Conviction": "ML model confidence (0-100%) that current data matches a historical 'Buy' pattern.",
}

# --- SIDEBAR & GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ System Control")
    is_mock = st.toggle("🛰️ Mock Data Mode", value=False)
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.session_state.market_data = {} 
        st.rerun()

# --- SMART FETCHING WRAPPER ---
@st.cache_data(ttl=3600)
def fetch_processed_data(ticker, use_mock=False):
    fetcher = DataFetcher(ticker)
    is_actual_mock = False
    
    if use_mock:
        hist_df = fetcher.generate_mock_data()
        is_actual_mock = True
    else:
        hist_df = fetcher.fetch_historical_data()
        if hist_df.empty:
            hist_df = fetcher.generate_mock_data()
            is_actual_mock = True
            
    return fetcher.add_indicators(hist_df), is_actual_mock

def get_data_buffered(ticker, use_mock):
    if "market_data" not in st.session_state:
        st.session_state.market_data = {}
    if ticker not in st.session_state.market_data:
        st.session_state.market_data[ticker] = fetch_processed_data(ticker, use_mock)
    return st.session_state.market_data[ticker]

# --- UTILITIES ---
def get_market_info(market_code):
    tz_map = {"US": "US/Eastern", "ES": "Europe/Madrid", "UK": "Europe/London"}
    tz = pytz.timezone(tz_map.get(market_code, "UTC"))
    now = datetime.now(tz)
    is_weekday = now.weekday() < 5
    is_hours = 9 <= now.hour < 17 or (now.hour == 17 and now.minute <= 30)
    status = "🟢 Open" if (is_weekday and is_hours) else "🔴 Closed"
    return status, now.strftime("%H:%M:%S")

@st.cache_resource
def load_assets():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_assets()

def create_price_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dash')))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    return fig

# --- SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🦅 Dip-Hunter Access")
        pwd_input = st.text_input("Enter credentials", type="password")
        if st.button("Login"):
            if pwd_input == st.secrets.get("password"):
                st.session_state["password_correct"] = True
                st.rerun()
            else: st.error("😕 Password incorrect")
        return False
    return True

# --- PAGE 1: WAR ROOM ---
def show_war_room():
    st.title("🦅 Trade AI: War Room")
    
    # Market Status Metrics
    m_cols = st.columns(3)
    for i, (m, label) in enumerate([("US", "Wall Street"), ("UK", "London"), ("ES", "Madrid")]):
        status, local_time = get_market_info(m)
        m_cols[i].metric(label, status, f"Local Time: {local_time}")
    
    st.markdown("---")
    
    dashboard_data, all_ticker_dfs = [], {}
    with st.spinner('Syncing Watchlist...'):
        for item in WATCHLIST:
            ticker = item["Ticker"]
            data, was_mocked = get_data_buffered(ticker, is_mock)
            if not data.empty:
                latest = data.iloc[-1]
                all_ticker_dfs[ticker] = data
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                dashboard_data.append({
                    "Ticker": ticker, 
                    "Price": latest['Close'], 
                    "RSI": latest['RSI'], 
                    "Conviction": prob,
                    "Change": ((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close']) if len(data) > 1 else 0
                })

    if dashboard_data:
        # --- NEW SUMMARY SECTION ---
        st.subheader("🎯 Actionable Signals")
        # Sort by conviction score (Descending)
        top_picks = sorted(dashboard_data, key=lambda x: x['Conviction'], reverse=True)[:3]
        
        s_cols = st.columns(len(top_picks))
        for i, pick in enumerate(top_picks):
            with s_cols[i]:
                color = "inverse" if pick['Conviction'] > 0.7 else "normal"
                st.metric(
                    label=f"TOP PICK: {pick['Ticker']}", 
                    value=f"{pick['Conviction']:.1%}", 
                    delta=f"{pick['Change']:.2%}",
                    help="Highest AI score based on technical pattern matching."
                )

        st.markdown("---")
        
        # Main Data Table
        st.dataframe(
            pd.DataFrame(dashboard_data).style.format({
                "Price": "{:.2f}", "Conviction": "{:.1%}", "RSI": "{:.1f}", "Change": "{:.2%}"
            }), 
            use_container_width=True, hide_index=True
        )
        
        # Details Tabs
        st.subheader("📊 Detailed Analysis")
        tabs = st.tabs([d["Ticker"] for d in dashboard_data])
        for i, tab in enumerate(tabs):
            with tab:
                row = dashboard_data[i]
                c1, c2 = st.columns([1, 3])
                c1.metric("Conviction", f"{row['Conviction']:.1%}")
                c1.metric("RSI", f"{row['RSI']:.1f}")
                c2.plotly_chart(create_price_chart(all_ticker_dfs[row["Ticker"]], row["Ticker"]))

# --- PAGE 2 & 3 (SCANNER/PORTFOLIO) ---
def show_scanner():
    st.title("🔍 AI Ticker Scanner")
    target = st.text_input("Enter Ticker").upper()
    if target:
        if st.button(f"Analyze {target}"):
            data, was_mocked = get_data_buffered(target, is_mock)
            if not data.empty:
                latest = data.iloc[-1]
                feat = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(feat)[0][1] if ml_model else 0.5
                st.metric("AI Conviction", f"{prob:.1%}")
                st.plotly_chart(create_price_chart(data, target))

def show_portfolio():
    st.title("💰 Live Portfolio")
    # ... Same Trading 212 logic from previous version ...
    st.write("Fetch Trading 212 positions and apply Risk Audit here.")

# --- NAVIGATION ---
if check_password():
    pg = st.navigation([
        st.Page(show_war_room, title="War Room", icon="🦅"),
        st.Page(show_scanner, title="AI Scanner", icon="🔍"),
        st.Page(show_portfolio, title="My Portfolio", icon="💰")
    ])
    pg.run()