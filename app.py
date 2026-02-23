import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from etl.fetcher import DataFetcher
import os
import time

# def get_market_status(market_code):
    # """Returns a status string and icon based on exchange hours."""
    # # Define timezones for your markets
    # tz_map = {
    #     "US": "US/Eastern",
    #     "ES": "Europe/Madrid",
    #     "UK": "Europe/London"
    # }
    
    # tz = pytz.timezone(tz_map.get(market_code, "UTC"))
    # now = datetime.now(tz)
    
    # # Standard Market Hours: 09:00 - 17:30
    # is_weekday = now.weekday() < 5
    # is_hours = 9 <= now.hour < 17 or (now.hour == 17 and now.minute <= 30)
    
    # if is_weekday and is_hours:
    #     return "🟢 Open", "Normal Trading"
    # elif is_weekday:
    #     return "🔴 Closed", "After Hours"
    # else:
    #     return "🔴 Closed", "Weekend"

# --- UI COMPONENT: TOP STATUS BAR ---
cols = st.columns(3)
with cols[0]:
    status, msg = get_market_status("US")
    st.metric("Wall Street (NYC)", status, msg)
with cols[1]:
    status, msg = get_market_status("UK")
    st.metric("London (LSE)", status, msg)
with cols[2]:
    status, msg = get_market_status("ES")
    st.metric("Madrid (BME)", status, msg)

# --- SECURITY: SIMPLE LOGIN ---
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

if not check_password():
    st.stop()

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=900)  # Cache results for 15 minutes (900 seconds)
def get_market_data(watchlist):
    """Fetches and processes data once, then stores it in memory."""
    dashboard_data = []
    all_ticker_dfs = {}
    
    for item in watchlist:
        ticker = item["Ticker"]
        fetcher = DataFetcher(ticker)
        
        # 1. Extraction with a small delay (1s) to respect Rate Limits
        hist_df = fetcher.fetch_historical_data(period="1y")
        full_data = fetcher.add_indicators(hist_df)
        
        if full_data.empty:
            continue
            
        latest = full_data.iloc[-1]
        all_ticker_dfs[ticker] = full_data
        
        # Determine ML Prediction (Moved inside cache to avoid re-calculation)
        # Note: ml_model is loaded globally below
        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
        buy_prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
        
        current_price = latest['Close']
        stop_buffer = current_price * (latest['Volatility'] * 2)
        
        dashboard_data.append({
            "Ticker": ticker,
            "Price": round(current_price, 2),
            "RSI": round(latest['RSI'], 1),
            "AI Conviction": buy_prob,
            "Stop Loss": round(current_price - stop_buffer, 2),
            "Target": round(current_price + (stop_buffer * 2), 2),
            "Vol": round(latest['Volatility'], 4)
        })
    return dashboard_data, all_ticker_dfs

# --- CONFIGURATION & MODELS ---
WATCHLIST = [
    {"Ticker": "CBRE",   "Market": "US"},
    {"Ticker": "SAN.MC", "Market": "ES"},
    {"Ticker": "RTO.L",  "Market": "UK"},
    {"Ticker": "CSCO",   "Market": "US"},
    {"Ticker": "GLEN.L", "Market": "UK"},
]

@st.cache_resource
def load_assets():
    try:
        return joblib.load('models/random_forest_model.pkl')
    except:
        return None

ml_model = load_assets()

# --- VISUALIZATION FUNCTIONS ---
def create_price_chart(df, ticker, buy_prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dash')))
    fig.update_layout(
        title=f"{ticker} Analysis (AI Conviction: {buy_prob:.1%})",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# --- APP LAYOUT ---
st.set_page_config(page_title="Trade AI: War Room", layout="wide")
st.title("🦅 Trade AI: Dip-Hunter Pro")

# Sidebar Logic
if st.sidebar.button('🔄 Clear Cache & Refresh'):
    st.cache_data.clear()
    st.rerun()

# --- EXECUTION ---
with st.spinner('Synchronizing Global Markets...'):
    dashboard_data, all_ticker_dfs = get_market_data(WATCHLIST)

if not dashboard_data:
    st.error("No data fetched. Yahoo Finance might be blocking the request. Try again in 10 minutes.")
    st.stop()

# --- UI COMPONENT: TOP TABLE ---
st.subheader("📡 Live Signals")
df_display = pd.DataFrame(dashboard_data)

def color_conviction(val):
    if val > 0.7: return 'background-color: #2ecc71'
    elif val < 0.3: return 'background-color: #e74c3c'
    return ''

st.dataframe(
    df_display.style.format({"AI Conviction": "{:.1%}"}).applymap(color_conviction, subset=['AI Conviction']),
    use_container_width=True,
    hide_index=True
)

# --- UI COMPONENT: DEEP DIVE VISUALS ---
st.markdown("---")
tabs = st.tabs([item["Ticker"] for item in WATCHLIST])

for i, tab in enumerate(tabs):
    ticker = WATCHLIST[i]["Ticker"]
    # Safety check if ticker failed to fetch
    if ticker not in all_ticker_dfs:
        continue
        
    row = next(item for item in dashboard_data if item["Ticker"] == ticker)
    
    with tab:
        col_text, col_chart = st.columns([1, 2])
        with col_text:
            st.metric("AI Conviction", f"{row['AI Conviction']:.1%}")
            st.write(f"**RSI:** {row['RSI']}")
            st.write(f"**Stop/Target:** {row['Stop Loss']} / {row['Target']}")
            
            if row['AI Conviction'] > 0.7:
                st.success("Bullish signal: Deep dip detected.")
            elif row['AI Conviction'] < 0.3:
                st.error("Bearish signal: Momentum failing.")
            else:
                st.warning("Neutral: Awaiting volume confirmation.")

        with col_chart:
            chart = create_price_chart(all_ticker_dfs[ticker], ticker, row['AI Conviction'])
            st.plotly_chart(chart, use_container_width=True)

st.caption("Engine: Random Forest v1.0 | Cache TTL: 15m")