import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import datetime
from etl.fetcher import DataFetcher, get_index_tickers

# --- CONFIG ---
st.set_page_config(page_title="Trade AI Pro", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_ml_model():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_ml_model()
fetcher_tool = DataFetcher()

# --- HELPER: SCORING ENGINE ---
def calculate_ai_score(df):
    if df.empty or ml_model is None:
        return 0.5
    try:
        # Ensure indicators are present
        df_ind = fetcher_tool.add_indicators(df)
        if df_ind.empty: return 0.5
        latest = df_ind.iloc[-1]
        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
        prob = ml_model.predict_proba(features)[0][1]
        return prob
    except:
        return 0.5

# --- PAGE: WAR ROOM ---
def show_war_room():
    st.title("🦅 Tactical War Room")
    st.markdown("### Top Market Signals")
    st.info("Run a 'Global Discovery' scan to populate the latest market opportunities.")
    # Here you could add logic to show the best results from the last scan saved in st.session_state

# --- PAGE: GLOBAL DISCOVERY ---
def show_global_discovery():
    st.title("🚀 Global AI Discovery")
    
    with st.expander("🔍 Filter Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        target_index = c1.selectbox("Market Index", ["SP500", "FTSE100", "IBEX35"])
        min_dollar_vol = c2.number_input("Min Daily Vol ($M)", value=1.0) * 1_000_000
        min_vol_thresh = c3.slider("Min Volatility (%)", 0.1, 5.0, 0.5) / 100

    if st.button("🔥 Start Market Scan"):
        tickers = get_index_tickers(target_index)
        if not tickers:
            st.error("No tickers found.")
            return

        with st.spinner(f"Scanning {len(tickers)} stocks..."):
            # Batch download
            data = yf.download(tickers, period="3mo", interval="1d", group_by='ticker', threads=True)
            
            results = []
            for ticker in tickers:
                try:
                    # Handle yfinance MultiIndex or SingleIndex
                    df = data[ticker].dropna() if len(tickers) > 1 else data.dropna()
                    if len(df) < 30: continue

                    # 1. LIQUIDITY CHECK
                    avg_vol = (df['Close'] * df['Volume']).tail(5).mean()
                    if avg_vol < min_dollar_vol: continue

                    # 2. VOLATILITY CHECK
                    vol = df['Close'].pct_change().tail(21).std()
                    if vol < min_vol_thresh: continue

                    # 3. AI SCORE
                    score = calculate_ai_score(df)

                    results.append({
                        "Ticker": ticker,
                        "AI Score": score,
                        "Price": df['Close'].iloc[-1],
                        "Volatility": f"{vol:.2%}",
                        "Liquidity ($M)": round(avg_vol / 1_000_000, 2)
                    })
                except: continue
            
            if results:
                df_res = pd.DataFrame(results).sort_values("AI Score", ascending=False)
                st.dataframe(df_res.style.background_gradient(subset=['AI Score'], cmap='RdYlGn'), use_container_width=True)
            else:
                st.warning("No stocks passed the filters. Try lowering the Volume requirement.")

# --- PAGE: TRADING 212 PORTFOLIO ---
def show_portfolio():
    st.title("💰 Trading 212 Portfolio")
    uploaded_file = st.file_uploader("Upload your T212 Export (CSV)", type="csv")
    
    if uploaded_file:
        portfolio_df = pd.read_csv(uploaded_file)
        # Attempt to find the ticker column (T212 labels it 'Ticker')
        ticker_col = 'Ticker' if 'Ticker' in portfolio_df.columns else None
        
        if ticker_col:
            unique_tickers = portfolio_df[ticker_col].unique().tolist()
            st.write(f"Analyzing {len(unique_tickers)} positions...")
            
            port_results = []
            for t in unique_tickers:
                # Basic cleaning for T212 tickers
                clean_t = str(t).split('_')[0] 
                df_hist = yf.Ticker(clean_t).history(period="6mo")
                score = calculate_ai_score(df_hist)
                
                port_results.append({
                    "Ticker": clean_t,
                    "AI Conviction": score,
                    "Current Price": df_hist['Close'].iloc[-1] if not df_hist.empty else 0
                })
            
            res_df = pd.DataFrame(port_results).sort_values("AI Conviction", ascending=False)
            st.table(res_df.style.background_gradient(subset=['AI Conviction'], cmap='RdYlGn'))
        else:
            st.error("Could not find a 'Ticker' column in the uploaded CSV.")

# --- NAVIGATION ---
pg = st.navigation([
    st.Page(show_war_room, title="War Room", icon="🦅"),
    st.Page(show_global_discovery, title="Global Discovery", icon="🚀"),
    st.Page(show_portfolio, title="Trading 212", icon="💰"),
])
pg.run()