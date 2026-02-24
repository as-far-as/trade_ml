import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import pytz
import base64
import requests
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
    if use_mock:
        hist_df = fetcher.generate_mock_data()
    else:
        hist_df = fetcher.fetch_historical_data()
        if hist_df.empty:
            st.toast(f"⚠️ API limit hit for {ticker}. Using Mock Data.")
            hist_df = fetcher.generate_mock_data()
    return fetcher.add_indicators(hist_df)

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
            data = fetch_processed_data(ticker, use_mock=is_mock)
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
    target = st.text_input("Enter Ticker (e.g. AAPL, MSFT, TSLA)").upper()
    if target:
        if st.button(f"Analyze {target}"):
            data = fetch_processed_data(target, use_mock=is_mock)
            if not data.empty:
                latest = data.iloc[-1]
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                c1, c2 = st.columns(2)
                c1.metric("AI Conviction", f"{prob:.1%}", help=TOOLTIPS["AI Conviction"])
                c2.metric("RSI", f"{latest['RSI']:.1f}")
                st.plotly_chart(create_price_chart(data, target))

# --- PAGE 3: PORTFOLIO ---
def show_portfolio():
    st.title("💰 My Live Portfolio")
    
    positions = get_t212_positions()
    
    if positions:
        portfolio_list = []
        for pos in positions:
            # 1. Correctly target the nested 'instrument' dictionary
            instrument = pos.get('instrument', {})
            raw_ticker = instrument.get('ticker', 'UNKNOWN')
            ticker = str(raw_ticker).split('_')[0] 
            
            # 2. Correctly target the nested 'walletImpact' dictionary
            wallet = pos.get('walletImpact', {})
            pnl_value = wallet.get('unrealizedProfitLoss', 0.0)
            pnl_currency = wallet.get('currency', '')
            
            portfolio_list.append({
                "Ticker": ticker,
                "Qty": pos.get('quantity', 0), 
                "Price": pos.get('currentPrice', 0), 
                "P&L": f"{pnl_value:.2f} {pnl_currency}".strip()
            })
        
        df_p = pd.DataFrame(portfolio_list)
        st.dataframe(df_p, use_container_width=True, hide_index=True)

        if st.button("Run AI Health Check"):
            for ticker in df_p['Ticker']:
                if ticker == "UNKNOWN": continue
                with st.expander(f"Audit: {ticker}"):
                    data = fetch_processed_data(ticker)
                    if not data.empty:
                        latest = data.iloc[-1]
                        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                        prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                        st.metric("AI Conviction", f"{prob:.1%}")
                        if prob < 0.3: st.error("🚨 Exit Signal: Poor Technical Alignment.")
                        elif prob > 0.7: st.success("💎 Hold: Strong Alignment.")
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