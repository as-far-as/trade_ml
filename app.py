import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import pytz
import numpy as np
from datetime import datetime
import requests
import base64
from etl.fetcher import DataFetcher

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Trade AI: War Room", layout="wide")

WATCHLIST = [
    {"Ticker": "CBRE",   "Market": "US"},
    {"Ticker": "SAN",    "Market": "US"}, 
    {"Ticker": "CSCO",   "Market": "US"},
]

# --- SIDEBAR & GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ System Control")
    is_mock = st.toggle(
        "🛰️ Mock Data Mode", 
        value=False, 
        help="Bypass Alpha Vantage and use synthetic data for testing/development."
    )
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption("Alpha Vantage Tier: Free (25 req/day)")

# --- SMART FETCHING WITH AUTO-FAILOVER ---
@st.cache_data(ttl=3600)
def fetch_processed_data(ticker, use_mock=False):
    """Fetches data with a safety net: API -> Cache -> Mock Failover."""
    fetcher = DataFetcher(ticker)
    
    if use_mock:
        hist_df = fetcher.generate_mock_data()
    else:
        hist_df = fetcher.fetch_historical_data()
        
        # If API returns empty (Limit hit or error), failover to Mock automatically
        if hist_df.empty:
            st.toast(f"⚠️ API Limit hit for {ticker}. Switching to Mock Data.", icon="🛑")
            hist_df = fetcher.generate_mock_data()
            
    return fetcher.add_indicators(hist_df)

# --- SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🦅 Dip-Hunter Access")
        if not st.secrets.get("password"):
            st.error("⚠️ Setup Error: No password found in secrets.toml.")
        
        pwd_input = st.text_input("Enter credentials", type="password")
        if st.button("Login"):
            if pwd_input == st.secrets.get("password"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Access Denied")
        return False
    return True

# --- TRADING 212 INTEGRATION ---
def get_t212_positions():
    key_id = st.secrets.get("t212_key_id", "").strip()
    secret = st.secrets.get("t212_secret", "").strip()
    if not key_id or not secret: return None

    auth_bytes = f"{key_id}:{secret}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_bytes).decode("utf-8")
    headers = {"Authorization": f"Basic {encoded_auth}", "Accept": "application/json"}
    
    try:
        res = requests.get("https://live.trading212.com/api/v0/equity/positions", headers=headers)
        return res.json() if res.status_code == 200 else None
    except:
        return None

# --- SHARED UTILITIES ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load('models/random_forest_model.pkl')
    except:
        return None

ml_model = load_assets()

def create_price_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dash')))
    fig.update_layout(
        template="plotly_dark", 
        height=400, 
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False
    )
    return fig

# --- PAGE: WAR ROOM ---
def show_war_room():
    st.title("🦅 Trade AI: War Room")
    if is_mock: st.info("📢 System running on Mock Data")
    st.markdown("---")
    
    dashboard_data = []
    with st.spinner('Syncing Market Data...'):
        for item in WATCHLIST:
            df = fetch_processed_data(item["Ticker"], use_mock=is_mock)
            if not df.empty:
                latest = df.iloc[-1]
                # ML Logic - ensure features match model training
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                
                dashboard_data.append({
                    "Ticker": item["Ticker"], 
                    "Price": f"${latest['Close']:.2f}", 
                    "RSI": round(latest['RSI'], 1), 
                    "AI Conviction": f"{prob:.1%}"
                })

    if dashboard_data:
        st.dataframe(pd.DataFrame(dashboard_data), use_container_width=True, hide_index=True)
        
        # Details Tabs
        tabs = st.tabs([d["Ticker"] for d in dashboard_data])
        for i, tab in enumerate(tabs):
            with tab:
                ticker = dashboard_data[i]["Ticker"]
                df = fetch_processed_data(ticker, use_mock=is_mock)
                st.plotly_chart(create_price_chart(df, ticker), use_container_width=True)

# --- PAGE: PORTFOLIO ---
def show_portfolio():
    st.title("💰 My Live Portfolio")
    positions = get_t212_positions()
    
    if positions:
        portfolio_list = []
        for pos in positions:
            instrument = pos.get('instrument', {})
            ticker = str(instrument.get('ticker', 'UNKNOWN')).split('_')[0] 
            wallet = pos.get('walletImpact', {})
            
            portfolio_list.append({
                "Ticker": ticker,
                "Qty": pos.get('quantity', 0), 
                "Price": pos.get('currentPrice', 0), 
                "P&L": wallet.get('unrealizedProfitLoss', 0.0),
                "Currency": wallet.get('currency', '')
            })
        
        df_p = pd.DataFrame(portfolio_list)
        
        # Styling the dataframe
        def color_pnl(val):
            color = '#00ff00' if val >= 0 else '#ff4b4b'
            return f'color: {color}'

        st.dataframe(
            df_p.style.applymap(color_pnl, subset=['P&L']).format({"P&L": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        if st.button("🛡️ Run AI Risk Audit"):
            for t in df_p['Ticker']:
                with st.expander(f"Audit Analysis: {t}"):
                    data = fetch_processed_data(t, use_mock=is_mock)
                    if not data.empty:
                        latest = data.iloc[-1]
                        feat = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                        prob = ml_model.predict_proba(feat)[0][1] if ml_model else 0.5
                        
                        c1, c2 = st.columns(2)
                        c1.metric("AI Health Score", f"{prob:.1%}")
                        c2.metric("RSI Level", f"{latest['RSI']:.1f}")
                        
                        if prob < 0.3: st.error("🚨 WARNING: Technical setup deteriorating.")
                        elif prob > 0.7: st.success("💎 STRONG HOLD: Pattern matches historical winners.")
    else:
        st.info("No open positions found or Trading 212 API connection failed.")

# --- NAVIGATION ---
if check_password():
    pg = st.navigation([
        st.Page(show_war_room, title="War Room", icon="🦅"),
        st.Page(show_portfolio, title="My Portfolio", icon="💰")
    ])
    pg.run()