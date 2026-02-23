import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from etl.fetcher import DataFetcher
import os
import time
import pytz
from datetime import datetime

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Trade AI: War Room", layout="wide")

WATCHLIST = [
    {"Ticker": "CBRE",   "Market": "US"},
    {"Ticker": "SAN.MC", "Market": "ES"},
    {"Ticker": "RTO.L",  "Market": "UK"},
    {"Ticker": "CSCO",   "Market": "US"},
    {"Ticker": "GLEN.L", "Market": "UK"},
]

# --- SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🦅 Dip-Hunter Access")
        st.text_input("Enter credentials", type="password", key="password_input")
        if st.button("Login"):
            if st.session_state["password_input"] == "trade_secret_2026":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Password incorrect")
        return False
    return True

# --- SHARED UTILITIES ---
@st.cache_resource
def load_assets():
    try:
        return joblib.load('models/random_forest_model.pkl')
    except:
        return None

ml_model = load_assets()

def get_market_status(market_code):
    tz_map = {"US": "US/Eastern", "ES": "Europe/Madrid", "UK": "Europe/London"}
    tz = pytz.timezone(tz_map.get(market_code, "UTC"))
    now = datetime.now(tz)
    is_weekday = now.weekday() < 5
    is_hours = 9 <= now.hour < 17 or (now.hour == 17 and now.minute <= 30)
    return ("🟢 Open", "Trading") if (is_weekday and is_hours) else ("🔴 Closed", "Market Idle")

def create_price_chart(df, ticker, buy_prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dash')))
    fig.update_layout(
        title=f"{ticker} Analysis (AI Conviction: {buy_prob:.1%})",
        template="plotly_dark", xaxis_rangeslider_visible=False, height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

@st.cache_data(ttl=900)
def fetch_processed_data(ticker):
    fetcher = DataFetcher(ticker)
    hist_df = fetcher.fetch_historical_data(period="1y")
    return fetcher.add_indicators(hist_df)

# --- PAGE 1: WAR ROOM ---
def show_war_room():
    st.title("🦅 Trade AI: War Room")
    
    # Market Status Bar
    m_cols = st.columns(3)
    for i, (m, label) in enumerate([("US", "Wall Street"), ("UK", "London"), ("ES", "Madrid")]):
        status, msg = get_market_status(m)
        m_cols[i].metric(label, status, msg)
    
    st.markdown("---")

    dashboard_data = []
    all_ticker_dfs = {}

    with st.spinner('Syncing Watchlist...'):
        for item in WATCHLIST:
            ticker = item["Ticker"]
            full_data = fetch_processed_data(ticker)
            if not full_data.empty:
                latest = full_data.iloc[-1]
                all_ticker_dfs[ticker] = full_data
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                buy_prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                
                dashboard_data.append({
                    "Ticker": ticker, "Price": latest['Close'], "RSI": latest['RSI'],
                    "AI Conviction": buy_prob, "Vol": latest['Volatility']
                })

    # Table
    df_display = pd.DataFrame(dashboard_data)
    st.dataframe(df_display.style.format({"Price": "{:.2f}", "AI Conviction": "{:.1%}"}), use_container_width=True, hide_index=True)

    # Deep Dive Tabs
    tabs = st.tabs([d["Ticker"] for d in dashboard_data])
    for i, tab in enumerate(tabs):
        ticker = dashboard_data[i]["Ticker"]
        with tab:
            st.plotly_chart(create_price_chart(all_ticker_dfs[ticker], ticker, dashboard_data[i]["AI Conviction"]))

# --- PAGE 2: SCANNER ---
def show_scanner():
    st.title("🔍 AI Ticker Scanner")
    st.write("Perform an on-demand AI audit on any global stock.")
    
    target = st.text_input("Enter Ticker (e.g. AAPL, TSLA, BTC-USD)", value="NVDA").upper()
    
    if st.button("Analyze Ticker"):
        with st.spinner(f"Running ML Audit on {target}..."):
            data = fetch_processed_data(target)
            if not data.empty:
                latest = data.iloc[-1]
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                
                c1, c2, c3 = st.columns(3)
                c1.metric("AI Conviction", f"{prob:.1%}")
                c2.metric("RSI", f"{latest['RSI']:.1f}")
                c3.metric("Volatility", f"{latest['Volatility']:.4f}")
                
                st.plotly_chart(create_price_chart(data, target, prob))
            else:
                st.error("Data fetch failed. Check the ticker symbol.")

# --- NAVIGATION ---
if check_password():
    pg = st.navigation([
        st.Page(show_war_room, title="War Room", icon="🦅"),
        st.Page(show_scanner, title="AI Scanner", icon="🔍"),
    ])
    pg.run()
    
    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.rerun()