import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from etl.fetcher import DataFetcher
import os
import time
import pytz
from datetime import datetime
import yfinance as yf

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Trade AI: War Room", layout="wide")

WATCHLIST = [
    {"Ticker": "CBRE",   "Market": "US"},
    {"Ticker": "SAN.MC", "Market": "ES"},
    {"Ticker": "RTO.L",  "Market": "UK"},
    {"Ticker": "CSCO",   "Market": "US"},
    {"Ticker": "GLEN.L", "Market": "UK"},
]

# --- TOOLTIP DEFINITIONS ---
TOOLTIPS = {
    "RSI": "Relative Strength Index (0-100). Below 30 is 'Oversold' (Buy opportunity), above 70 is 'Overbought' (Sell warning).",
    "AI Conviction": "The ML model's confidence (0-100%) that the current setup matches a historical 'Buy' pattern.",
    "Vol": "Volatility (Standard Deviation of returns). High values mean the price is swinging wildly.",
    "MA50": "The average price over the last 50 days. Acts as a support/resistance line."
}

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

def get_market_info(market_code):
    """Returns status and formatted current time for the market."""
    tz_map = {"US": "US/Eastern", "ES": "Europe/Madrid", "UK": "Europe/London"}
    tz = pytz.timezone(tz_map.get(market_code, "UTC"))
    now = datetime.now(tz)
    
    is_weekday = now.weekday() < 5
    is_hours = 9 <= now.hour < 17 or (now.hour == 17 and now.minute <= 30)
    status = "🟢 Open" if (is_weekday and is_hours) else "🔴 Closed"
    return status, now.strftime("%H:%M:%S")

def create_price_chart(df, ticker, buy_prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dash')))
    fig.update_layout(
        template="plotly_dark", xaxis_rangeslider_visible=False, height=400,
        margin=dict(l=10, r=10, t=30, b=10)
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
    
    # Market Status Bar with Time
    m_cols = st.columns(3)
    for i, (m, label) in enumerate([("US", "Wall Street"), ("UK", "London"), ("ES", "Madrid")]):
        status, local_time = get_market_info(m)
        m_cols[i].metric(label, status, f"Local Time: {local_time}")
    
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

    # Table with Tooltips in column headers
    df_display = pd.DataFrame(dashboard_data)
    st.dataframe(
        df_display.style.format({"Price": "{:.2f}", "AI Conviction": "{:.1%}", "RSI": "{:.1f}", "Vol": "{:.4f}"}), 
        use_container_width=True, hide_index=True
    )

    # Deep Dive Tabs
    tabs = st.tabs([d["Ticker"] for d in dashboard_data])
    for i, tab in enumerate(tabs):
        ticker = dashboard_data[i]["Ticker"]
        row = dashboard_data[i]
        with tab:
            c1, c2 = st.columns([1, 3])
            with c1:
                st.metric("AI Conviction", f"{row['AI Conviction']:.1%}", help=TOOLTIPS["AI Conviction"])
                st.metric("RSI", f"{row['RSI']:.1f}", help=TOOLTIPS["RSI"])
                st.metric("Volatility", f"{row['Vol']:.4f}", help=TOOLTIPS["Vol"])
            with c2:
                st.plotly_chart(create_price_chart(all_ticker_dfs[ticker], ticker, row["AI Conviction"]))

# --- PAGE 2: SCANNER ---
def show_scanner():
    st.title("🔍 AI Ticker Scanner")
    
    # ADVANCED SEARCH LOGIC
    search_query = st.text_input("Enter Company Name or Ticker (e.g. 'Apple' or 'AAPL')")
    
    if search_query:
        # Resolve Ticker using yfinance search
        search_results = yf.Search(search_query, max_results=5).quotes
        if not search_results:
            st.error("No results found.")
        else:
            # Let user pick the best match
            options = {f"{r['symbol']} ({r.get('shortname', 'N/A')})": r['symbol'] for r in search_results}
            selected_display = st.selectbox("Select the correct match:", list(options.keys()))
            target = options[selected_display]

            if st.button(f"Analyze {target}"):
                with st.spinner(f"Running ML Audit on {target}..."):
                    data = fetch_processed_data(target)
                    if not data.empty:
                        latest = data.iloc[-1]
                        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                        prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("AI Conviction", f"{prob:.1%}", help=TOOLTIPS["AI Conviction"])
                        c2.metric("RSI", f"{latest['RSI']:.1f}", help=TOOLTIPS["RSI"])
                        c3.metric("Volatility", f"{latest['Volatility']:.4f}", help=TOOLTIPS["Vol"])
                        
                        st.plotly_chart(create_price_chart(data, target, prob))
                    else:
                        st.error("Data fetch failed for this ticker.")

# --- NAVIGATION ---
if check_password():
    pg = st.navigation([
        st.Page(show_war_room, title="War Room", icon="🦅"),
        st.Page(show_scanner, title="AI Scanner", icon="🔍"),
    ])
    pg.run()