import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import datetime
# Import the standalone function and the class
from etl.fetcher import DataFetcher, get_index_tickers

st.set_page_config(page_title="Trade AI Pro", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_ml_model():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_ml_model()

# --- PAGE 1: WAR ROOM ---
def show_war_room():
    st.title("🦅 Tactical War Room")
    st.info("Market-wide scans from Global Discovery will appear here as top recommendations.")
    # (Your existing watchlist/summary logic remains here)

# --- PAGE 2: GLOBAL DISCOVERY (With Advanced Filters) ---
def show_global_discovery():
    st.title("🚀 Global AI Discovery")
    st.markdown("Scan entire indices for high-conviction setups.")

    # Sidebar / Controls
    with st.expander("🔍 Filter Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        target_index = c1.selectbox("Market Index", ["SP500", "FTSE100", "IBEX35"])
        # Set default to $10M for liquidity
        min_dollar_vol = c2.number_input("Min Daily Dollar Vol ($M)", value=10.0) * 1_000_000
        # Set default to 1.5% for "tradeable" movement
        min_vol_thresh = c3.slider("Min Volatility (Daily %)", 0.5, 5.0, 1.5) / 100

    if st.button("🔥 Start Market Scan"):
        # 1. Fetch Tickers (Bulletproof standalone call)
        tickers = get_index_tickers(target_index)
        
        if not tickers:
            st.error("Could not retrieve tickers. Check your connection.")
            return

        with st.spinner(f"Analyzing {len(tickers)} stocks... This takes ~30 seconds."):
            # 2. Batch Download (Last 6 months for indicator calculation)
            raw_data = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', threads=True)
            
            discovery_results = []
            progress = st.progress(0)
            
            for i, ticker in enumerate(tickers):
                try:
                    df = raw_data[ticker].dropna()
                    if len(df) < 50: continue

                    # 3. APPLY QUANT FILTERS
                    # Liquidity: Avg Dollar Volume over last 10 days
                    avg_liq = (df['Close'] * df['Volume']).tail(10).mean()
                    if avg_liq < min_dollar_vol: continue

                    # Volatility: Std Dev of returns over last 21 days
                    current_vol = df['Close'].pct_change().tail(21).std()
                    if current_vol < min_vol_thresh: continue

                    # 4. AI SCORING
                    f = DataFetcher(ticker)
                    df_ready = f.add_indicators(df)
                    if df_ready.empty: continue
                    
                    latest = df_ready.iloc[-1]
                    feats = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                    prob = ml_model.predict_proba(feats)[0][1] if ml_model else 0.5

                    discovery_results.append({
                        "Ticker": ticker,
                        "AI Score": prob,
                        "Price": latest['Close'],
                        "RSI": latest['RSI'],
                        "Volatility": f"{current_vol:.2%}",
                        "Daily Liq ($)": f"${avg_liq/1e6:.1f}M"
                    })
                except: continue
                progress.progress((i + 1) / len(tickers))

        if discovery_results:
            res_df = pd.DataFrame(discovery_results).sort_values("AI Score", ascending=False)
            
            # Highlight Top 3
            st.subheader("🎯 High Conviction Signals")
            top_cols = st.columns(3)
            for idx, row in res_df.head(3).iterrows():
                with top_cols[discovery_results.index(row)]: # Simple col mapping
                    st.metric(row['Ticker'], f"${row['Price']:.2f}", f"Score: {row['AI Score']:.1%}")

            st.dataframe(
                res_df.style.background_gradient(subset=['AI Score'], cmap='RdYlGn'),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("No stocks passed your filters. Try relaxing the Volume or Volatility requirements.")

# --- NAVIGATION ---
pg = st.navigation([
    st.Page(show_war_room, title="War Room", icon="🦅"),
    st.Page(show_global_discovery, title="Global Discovery", icon="🚀"),
])
pg.run()