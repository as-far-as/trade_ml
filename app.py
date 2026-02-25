import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import requests
from datetime import datetime
# Fixed Import: Ensures the etl folder is recognized
from etl.fetcher import DataFetcher, get_index_tickers

st.set_page_config(page_title="Trade AI Pro", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_ml_model():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_ml_model()
fetcher_tool = DataFetcher()

# --- T212 API LOGIC (Restored) ---
def fetch_t212_positions():
    api_key = st.secrets.get("T212_API_KEY")
    if not api_key:
        st.error("Missing T212_API_KEY in Streamlit Secrets.")
        return []
    
    # Trading 212 standard portfolio endpoint
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    headers = {"Authorization": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        st.error(f"T212 API Error: {response.status_code}")
        return []
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

def calculate_ai_score(df):
    if df is None or df.empty or ml_model is None: return 0.5
    try:
        df_ind = fetcher_tool.add_indicators(df)
        if df_ind.empty: return 0.5
        latest = df_ind.iloc[-1]
        # Aligning with your model's expected features
        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
        return ml_model.predict_proba(features)[0][1]
    except: return 0.5

# --- PAGE: GLOBAL DISCOVERY ---
def show_global_discovery():
    st.title("🚀 Global AI Discovery")
    
    with st.sidebar:
        st.header("Scan Settings")
        target_index = st.selectbox("Market Index", ["SP500", "FTSE100", "IBEX35"])
        # Defaults lowered to ensure you see results initially
        min_liq = st.number_input("Min Daily Vol ($M)", value=0.1) * 1_000_000
        min_vol = st.slider("Min Volatility (%)", 0.0, 5.0, 0.1) / 100

    if st.button("🔥 Run Market Scan"):
        tickers = get_index_tickers(target_index)
        if not tickers: return

        with st.spinner(f"Analyzing {len(tickers)} assets..."):
            # download with multi_level_index=False avoids the MultiIndex headache
            data = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', threads=True)
            
            discovery_results = []
            for ticker in tickers:
                try:
                    # Robust check for ticker data
                    if ticker not in data.columns.levels[0]: continue
                    df = data[ticker].dropna()
                    if len(df) < 50: continue

                    # Quant Filters
                    current_liq = (df['Close'] * df['Volume']).tail(5).mean()
                    volatility = df['Close'].pct_change().tail(21).std()

                    if current_liq >= min_liq and volatility >= min_vol:
                        score = calculate_ai_score(df)
                        discovery_results.append({
                            "Ticker": ticker,
                            "AI Score": score,
                            "Price": round(df['Close'].iloc[-1], 2),
                            "Vol%": f"{volatility:.2%}",
                            "Liq($M)": round(current_liq / 1_000_000, 2)
                        })
                except: continue

            if discovery_results:
                res_df = pd.DataFrame(discovery_results).sort_values("AI Score", ascending=False)
                st.dataframe(res_df.style.background_gradient(subset=['AI Score'], cmap='RdYlGn'), use_container_width=True)
            else:
                st.warning("No stocks passed. Try lowering the 'Min Daily Vol' filter.")

# --- PAGE: TRADING 212 ---
def show_portfolio():
    st.title("💰 Trading 212 Live Feed")
    if st.button("🔄 Sync Portfolio"):
        positions = fetch_t212_positions()
        
        if positions:
            results = []
            for pos in positions:
                ticker = pos['ticker'].split('_')[0] # Clean T212 suffix
                hist = yf.download(ticker, period="6mo", progress=False)
                score = calculate_ai_score(hist)
                results.append({
                    "Ticker": ticker,
                    "AI Score": score,
                    "Quantity": pos['quantity'],
                    "P&L": pos.get('ppl', 0)
                })
            
            st.table(pd.DataFrame(results).sort_values("AI Score", ascending=False).style.background_gradient(subset=['AI Score'], cmap='RdYlGn'))

# --- NAV ---
pg = st.navigation({
    "Scanner": [st.Page(show_global_discovery, title="Global Discovery", icon="🚀")],
    "Account": [st.Page(show_portfolio, title="Trading 212", icon="💰")]
})
pg.run()