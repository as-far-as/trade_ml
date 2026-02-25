import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from etl.fetcher import DataFetcher, get_index_tickers

# --- CONFIG ---
st.set_page_config(page_title="Trade AI Pro", layout="wide")
fetcher_tool = DataFetcher()

# --- ASSET LOADING ---
@st.cache_resource
def load_ml_model():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_ml_model()

# --- T212 API INTEGRATION ---
def fetch_t212_positions():
    """Fetches live positions using your T212 API Key."""
    api_key = st.secrets.get("T212_API_KEY")
    if not api_key:
        st.error("T212_API_KEY missing in secrets!")
        return []
    
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    headers = {"Authorization": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return []

def calculate_ai_score(df):
    if df.empty or ml_model is None: return 0.5
    try:
        df_ind = fetcher_tool.add_indicators(df)
        if df_ind.empty: return 0.5
        latest = df_ind.iloc[-1]
        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
        return ml_model.predict_proba(features)[0][1]
    except: return 0.5

# --- PAGE: GLOBAL DISCOVERY ---
def show_global_discovery():
    st.title("🚀 Global AI Discovery")
    
    with st.expander("🔍 Market Scanning Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        target_index = c1.selectbox("Market Index", ["SP500", "FTSE100", "IBEX35"])
        # Loosened default filters to ensure results populate
        min_dollar_vol = c2.number_input("Min Daily Vol ($M)", value=0.5) * 1_000_000
        min_vol_thresh = c3.slider("Min Volatility (%)", 0.0, 5.0, 0.2) / 100

    if st.button("🔥 Start Market Scan"):
        tickers = get_index_tickers(target_index)
        if not tickers: return

        with st.spinner(f"Scanning {len(tickers)} stocks..."):
            # Batch download (Fixed with threads=True)
            data = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', threads=True)
            
            results = []
            for ticker in tickers:
                try:
                    # Fix for MultiIndex: ensures we get the sub-dataframe for the ticker
                    df = data[ticker].dropna() if ticker in data.columns.levels[0] else pd.DataFrame()
                    if len(df) < 50: continue

                    # CALCULATE FILTERS
                    # Using tail(5) for recent liquidity
                    avg_liq = (df['Close'] * df['Volume']).tail(5).mean()
                    current_vol = df['Close'].pct_change().tail(21).std()

                    if avg_liq >= min_dollar_vol and current_vol >= min_vol_thresh:
                        score = calculate_ai_score(df)
                        results.append({
                            "Ticker": ticker,
                            "AI Score": score,
                            "Price": df['Close'].iloc[-1],
                            "Volatility": f"{current_vol:.2%}",
                            "Liquidity ($M)": round(avg_liq / 1_000_000, 2)
                        })
                except: continue
            
            if results:
                df_res = pd.DataFrame(results).sort_values("AI Score", ascending=False)
                st.session_state['last_scan'] = df_res # Save for War Room
                st.dataframe(df_res.style.background_gradient(subset=['AI Score'], cmap='RdYlGn'), use_container_width=True)
            else:
                st.warning("No stocks passed the filters. Try lowering the requirements.")

# --- PAGE: TRADING 212 (API VERSION) ---
def show_portfolio():
    st.title("💰 Live Trading 212 Portfolio")
    
    if st.button("🔄 Sync with T212"):
        positions = fetch_t212_positions()
        if not positions:
            st.warning("No active positions found or API error.")
            return

        port_results = []
        with st.spinner("Analyzing portfolio health..."):
            for pos in positions:
                ticker = pos['ticker']
                # T212 usually uses 'TICKER_US_EQ' - we need to clean this for Yahoo
                clean_ticker = ticker.split('_')[0]
                
                df_hist = yf.download(clean_ticker, period="6mo", progress=False)
                score = calculate_ai_score(df_hist)
                
                port_results.append({
                    "Ticker": clean_ticker,
                    "Quantity": pos['quantity'],
                    "AI Conviction": score,
                    "Current Price": df_hist['Close'].iloc[-1] if not df_hist.empty else 0,
                    "Result": pos.get('ppl', 0) # Profit/Loss
                })
        
        res_df = pd.DataFrame(port_results).sort_values("AI Conviction", ascending=False)
        st.table(res_df.style.background_gradient(subset=['AI Conviction'], cmap='RdYlGn'))

# --- NAVIGATION ---
pg = st.navigation([
    st.Page(show_global_discovery, title="Global Discovery", icon="🚀"),
    st.Page(show_portfolio, title="Trading 212", icon="💰"),
])
pg.run()