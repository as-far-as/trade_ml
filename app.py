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
    "Vol": "Volatility (Std Dev of returns). High values indicate large price swings.",
    "MA50": "The 50-day Moving Average trend line."
}

# --- SIDEBAR & GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ System Control")
    is_mock = st.toggle("🛰️ Mock Data Mode", value=False, help="Use synthetic data if API is down.")
    if st.button("Clear Cache & Refresh"):
        st.cache_data.clear()
        st.rerun()

# --- SMART FETCHING WRAPPER ---
@st.cache_data(ttl=3600)
def fetch_processed_data(ticker, use_mock=False):
    fetcher = DataFetcher(ticker)
    is_actual_mock = False # Track if we actually used mock data
    
    if use_mock:
        hist_df = fetcher.generate_mock_data()
        is_actual_mock = True
    else:
        hist_df = fetcher.fetch_historical_data()
        if hist_df.empty:
            hist_df = fetcher.generate_mock_data()
            is_actual_mock = True
            
    # Return both the data AND a flag indicating if it's mock data
    return fetcher.add_indicators(hist_df), is_actual_mock

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
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
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

# --- TRADING 212 INTEGRATION ---
def get_t212_positions():
    key_id = st.secrets.get("t212_key_id", "").strip()
    secret = st.secrets.get("t212_secret", "").strip()
    if not key_id or not secret:
        st.error("🔑 Secrets missing! Check your t212_key_id and t212_secret in Streamlit.")
        return None

    auth_bytes = f"{key_id}:{secret}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_bytes).decode("utf-8")
    headers = {"Authorization": f"Basic {encoded_auth}", "Accept": "application/json"}
    
    url = "https://live.trading212.com/api/v0/equity/positions"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Network Error: {e}")
        return None

# --- PAGE 1: WAR ROOM ---
def show_war_room():
    st.title("🦅 Trade AI: War Room")
    m_cols = st.columns(3)
    for i, (m, label) in enumerate([("US", "Wall Street"), ("UK", "London"), ("ES", "Madrid")]):
        status, local_time = get_market_info(m)
        m_cols[i].metric(label, status, f"Local Time: {local_time}")
    
    st.markdown("---")
    dashboard_data, all_ticker_dfs = [], {}

    with st.spinner('Syncing Watchlist...'):
        for item in WATCHLIST:
            ticker = item["Ticker"]
            data, was_mocked = fetch_processed_data(ticker, use_mock=is_mock)
            
            if was_mocked and not is_mock:
                st.toast(f"⚠️ API limit hit for {ticker}. Using Mock Data.", icon="🛑")
            
            if not data.empty:
                latest = data.iloc[-1]
                all_ticker_dfs[ticker] = data
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                dashboard_data.append({"Ticker": ticker, "Price": latest['Close'], "RSI": latest['RSI'], "AI Conviction": prob, "Vol": latest['Volatility']})

    if dashboard_data:
        df_disp = pd.DataFrame(dashboard_data)
        st.dataframe(df_disp.style.format({"Price": "{:.2f}", "AI Conviction": "{:.1%}", "RSI": "{:.1f}", "Vol": "{:.4f}"}), use_container_width=True, hide_index=True)
        
        tabs = st.tabs([d["Ticker"] for d in dashboard_data])
        for i, tab in enumerate(tabs):
            with tab:
                row = dashboard_data[i]
                c1, c2 = st.columns([1, 3])
                c1.metric("AI Conviction", f"{row['AI Conviction']:.1%}", help=TOOLTIPS["AI Conviction"])
                c1.metric("RSI", f"{row['RSI']:.1f}", help=TOOLTIPS["RSI"])
                c2.plotly_chart(create_price_chart(all_ticker_dfs[row["Ticker"]], row["Ticker"]))

# --- PAGE 2: SCANNER ---
def show_scanner():
    st.title("🔍 AI Ticker Scanner")
    target = st.text_input("Enter Ticker (e.g. AAPL, MSFT)").upper()
    if target:
        if st.button(f"Analyze {target}"):
            data = fetch_processed_data(target, use_mock=is_mock)
            if not data.empty:
                latest = data.iloc[-1]
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                c1, c2 = st.columns(2)
                c1.metric("AI Conviction", f"{prob:.1%}")
                c2.metric("RSI", f"{latest['RSI']:.1f}")
                st.plotly_chart(create_price_chart(data, target))

# --- PAGE 3: PORTFOLIO ---
def show_portfolio():
    st.title("💰 Live Portfolio")
    positions = get_t212_positions()
    
    if positions:
        portfolio_list = []
        for pos in positions:
            instrument = pos.get('instrument', {})
            ticker = str(instrument.get('ticker', 'UNKNOWN')).split('_')[0] 
            wallet = pos.get('walletImpact', {})
            pnl_val = wallet.get('unrealizedProfitLoss', 0.0)
            
            portfolio_list.append({
                "Ticker": ticker,
                "Qty": pos.get('quantity', 0), 
                "Price": pos.get('currentPrice', 0), 
                "P&L": pnl_val,
                "Cur": wallet.get('currency', '')
            })
        
        df_p = pd.DataFrame(portfolio_list)
        
        def color_pnl(val):
            color = '#00ff00' if val >= 0 else '#ff4b4b'
            return f'color: {color}'

        st.dataframe(
            df_p.style.applymap(color_pnl, subset=['P&L']).format({"P&L": "{:.2f}"}),
            use_container_width=True, hide_index=True
        )

        if st.button("🛡️ Run AI Risk Audit"):
            for t in df_p['Ticker']:
                with st.expander(f"Audit: {t}"):
                    data = fetch_processed_data(t, use_mock=is_mock)
                    if not data.empty:
                        latest = data.iloc[-1]
                        feat = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                        prob = ml_model.predict_proba(feat)[0][1] if ml_model else 0.5
                        st.metric("AI Conviction", f"{prob:.1%}")
    else:
        st.info("No open positions found or connection failed.")

# --- NAVIGATION ---
if check_password():
    pg = st.navigation([
        st.Page(show_war_room, title="War Room", icon="🦅"),
        st.Page(show_scanner, title="AI Scanner", icon="🔍"),
        st.Page(show_portfolio, title="My Portfolio", icon="💰")
    ])
    pg.run()