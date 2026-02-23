import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from etl.fetcher import DataFetcher
import os
import pytz
from datetime import datetime
import yfinance as yf
import requests
import base64

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Trade AI: War Room", layout="wide")

WATCHLIST = [
    {"Ticker": "CBRE",   "Market": "US"},
    {"Ticker": "SAN.MC", "Market": "ES"},
    {"Ticker": "RTO.L",  "Market": "UK"},
    {"Ticker": "CSCO",   "Market": "US"},
    {"Ticker": "GLEN.L", "Market": "UK"},
]

TOOLTIPS = {
    "RSI": "Relative Strength Index (0-100). Below 30 is 'Oversold', above 70 is 'Overbought'.",
    "AI Conviction": "ML model confidence (0-100%) that current data matches a historical 'Buy' pattern.",
    "Vol": "Volatility (Std Dev of returns). High values indicate large price swings.",
    "MA50": "The 50-day Moving Average trend line."
}

# --- SECURITY & DEBUG ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🦅 Dip-Hunter Access")
        
        # DEBUG SECTION: Shows if secrets are loaded
        if not st.secrets:
            st.error("⚠️ No secrets.toml detected. Check your .streamlit folder.")
        
        pwd_input = st.text_input("Enter credentials", type="password")
        if st.button("Login"):
            if pwd_input == st.secrets.get("password"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Password incorrect")
        return False
    return True

# --- TRADING 212 API (BASIC AUTH) ---
def get_t212_positions():
    # Attempt to pull from secrets
    key_id = st.secrets.get("t212_key_id", "").strip()
    secret = st.secrets.get("t212_secret", "").strip()
    
    if not key_id or not secret:
        st.error("🔑 Secrets missing! Check your .streamlit/secrets.toml file structure.")
        return None

    # Basic Auth Encoding
    auth_bytes = f"{key_id}:{secret}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_bytes).decode("utf-8")
    
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Accept": "application/json"
    }
    
    # Try LIVE first
    url = "https://live.trading212.com/api/v0/equity/positions"
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 401:
            # Automatic fallback check for Demo if Live fails
            demo_url = "https://demo.trading212.com/api/v0/equity/positions"
            demo_res = requests.get(demo_url, headers=headers)
            if demo_res.status_code == 200:
                st.warning("⚠️ Your keys worked on the DEMO server, but not LIVE. Switching to Demo data.")
                return demo_res.json()
            
            st.error("🚫 401: Both Live and Demo rejected these keys. Please re-generate them in T212.")
            return None
            
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Network Error: {e}")
        return None

# --- SHARED UTILITIES ---
@st.cache_resource
def load_assets():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_assets()

def get_market_info(market_code):
    tz_map = {"US": "US/Eastern", "ES": "Europe/Madrid", "UK": "Europe/London"}
    tz = pytz.timezone(tz_map.get(market_code, "UTC"))
    now = datetime.now(tz)
    is_weekday = now.weekday() < 5
    is_hours = 9 <= now.hour < 17 or (now.hour == 17 and now.minute <= 30)
    status = "🟢 Open" if (is_weekday and is_hours) else "🔴 Closed"
    return status, now.strftime("%H:%M:%S")

@st.cache_data(ttl=900)
def fetch_processed_data(ticker):
    fetcher = DataFetcher(ticker)
    hist_df = fetcher.fetch_historical_data(period="1y")
    return fetcher.add_indicators(hist_df)

def create_price_chart(df, ticker, buy_prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dash')))
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=400, margin=dict(l=10, r=10, t=30, b=10))
    return fig

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
            data = fetch_processed_data(ticker)
            if not data.empty:
                latest = data.iloc[-1]
                all_ticker_dfs[ticker] = data
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                dashboard_data.append({"Ticker": ticker, "Price": latest['Close'], "RSI": latest['RSI'], "AI Conviction": prob, "Vol": latest['Volatility']})

    df_disp = pd.DataFrame(dashboard_data)
    st.dataframe(df_disp.style.format({"Price": "{:.2f}", "AI Conviction": "{:.1%}", "RSI": "{:.1f}", "Vol": "{:.4f}"}), use_container_width=True, hide_index=True)
    
    tabs = st.tabs([d["Ticker"] for d in dashboard_data])
    for i, tab in enumerate(tabs):
        with tab:
            row = dashboard_data[i]
            c1, c2 = st.columns([1, 3])
            c1.metric("AI Conviction", f"{row['AI Conviction']:.1%}", help=TOOLTIPS["AI Conviction"])
            c1.metric("RSI", f"{row['RSI']:.1f}", help=TOOLTIPS["RSI"])
            c2.plotly_chart(create_price_chart(all_ticker_dfs[row["Ticker"]], row["Ticker"], row["AI Conviction"]))

# --- PAGE 2: SCANNER ---
def show_scanner():
    st.title("🔍 AI Ticker Scanner")
    search_query = st.text_input("Enter Company Name or Ticker")
    if search_query:
        results = yf.Search(search_query, max_results=5).quotes
        if results:
            options = {f"{r['symbol']} ({r.get('shortname', 'N/A')})": r['symbol'] for r in results}
            target = options[st.selectbox("Select Match:", list(options.keys()))]
            if st.button(f"Analyze {target}"):
                data = fetch_processed_data(target)
                if not data.empty:
                    latest = data.iloc[-1]
                    features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                    prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                    c1, c2 = st.columns(2)
                    c1.metric("AI Conviction", f"{prob:.1%}", help=TOOLTIPS["AI Conviction"])
                    c2.metric("RSI", f"{latest['RSI']:.1f}")
                    st.plotly_chart(create_price_chart(data, target, prob))

# --- PAGE 3: PORTFOLIO ---
def show_portfolio():
    st.title("💰 My Live Portfolio")
    
    # Verify both keys exist in secrets
    if "t212_key_id" not in st.secrets or "t212_secret" not in st.secrets:
        st.warning("⚠️ No T212 credentials found. Check your secrets.toml file.")
        return

    positions = get_t212_positions()
    if positions:
        st.success(f"Synced {len(positions)} positions.")
        portfolio_list = []
        for pos in positions:
            ticker = pos['ticker'].split('_')[0] 
            portfolio_list.append({
                "Ticker": ticker, "Qty": pos['quantity'], 
                "Price": pos['currentPrice'], "P&L": f"{pos['ppl']:.2f} {pos['currency']}"
            })
        
        df_p = pd.DataFrame(portfolio_list)
        st.table(df_p)

        if st.button("Run AI Health Check"):
            for ticker in df_p['Ticker']:
                with st.expander(f"Audit: {ticker}"):
                    data = fetch_processed_data(ticker)
                    if not data.empty:
                        latest = data.iloc[-1]
                        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                        prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                        st.metric("AI Conviction", f"{prob:.1%}")
                        if prob < 0.3: st.error("🚨 Exit Signal: Poor Technical Setup.")
                        elif prob > 0.7: st.success("💎 Hold: Strong Alignment.")

# --- NAVIGATION ---
if check_password():
    pg = st.navigation([
        st.Page(show_war_room, title="War Room", icon="🦅"),
        st.Page(show_scanner, title="AI Scanner", icon="🔍"),
        st.Page(show_portfolio, title="My Portfolio", icon="💰")
    ])
    pg.run()
    
    if st.sidebar.button("Refresh Cache"):
        st.cache_data.clear()
        st.rerun()