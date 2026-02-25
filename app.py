import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.graph_objects as go
from datetime import datetime
from etl.fetcher import DataFetcher

# --- APP CONFIG ---
st.set_page_config(page_title="Trade AI Pro", layout="wide", initial_sidebar_state="expanded")
fetcher = DataFetcher()

# Load ML Model
@st.cache_resource
def load_ml_model():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_ml_model()

# --- REUSABLE COMPONENTS ---
def create_mini_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], line=dict(color='#00d4ff', width=2)))
    fig.update_layout(template="plotly_dark", height=100, margin=dict(l=0,r=0,t=0,b=0), 
                      xaxis_visible=False, yaxis_visible=False, showlegend=False)
    return fig

# --- PAGE: GLOBAL DISCOVERY ---
def show_global_discovery():
    st.title("🚀 AI Global Discovery")
    st.markdown("Scan entire indices for high-conviction AI setups based on volume and volatility filters.")

    c1, c2, c3 = st.columns([2, 1, 1])
    target_index = c1.selectbox("Select Market", ["SP500", "FTSE100", "IBEX35"])
    min_volume = c2.number_input("Min Daily Vol ($M)", value=10.0) * 1_000_000
    min_volatility = c3.slider("Min Volatility (%)", 0.5, 5.0, 1.0) / 100

    if st.button("🔥 Run Market-Wide Scan"):
        tickers = fetcher.get_index_tickers(target_index)
        
        with st.spinner(f"Downloading data for {len(tickers)} tickers..."):
            # Batch download 6 months of data
            raw_data = yf.download(tickers, period="6mo", interval="1d", group_by='ticker', threads=True)
            
        processed_results = []
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                df = raw_data[ticker].dropna()
                if len(df) < 50: continue

                # 1. LIQUIDITY FILTER
                avg_dollar_vol = (df['Close'] * df['Volume']).tail(10).mean()
                if avg_dollar_vol < min_volume: continue

                # 2. VOLATILITY FILTER
                current_vol = df['Close'].pct_change().tail(21).std()
                if current_vol < min_volatility: continue

                # 3. AI SCORING
                df_with_inds = fetcher.add_indicators(df)
                if df_with_inds.empty: continue
                
                latest = df_with_inds.iloc[-1]
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5

                processed_results.append({
                    "Ticker": ticker,
                    "Price": latest['Close'],
                    "AI Score": prob,
                    "RSI": latest['RSI'],
                    "Volatiltiy": current_vol,
                    "Daily Vol ($)": avg_dollar_vol
                })
            except: continue
            progress_bar.progress((i + 1) / len(tickers))

        if processed_results:
            results_df = pd.DataFrame(processed_results).sort_values("AI Score", ascending=False)
            
            st.subheader(f"Top AI Picks in {target_index}")
            # Display Top 3 in Cards
            top_3 = results_df.head(3)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.metric(row['Ticker'], f"${row['Price']:.2f}", f"Score: {row['AI Score']:.1%}")
            
            st.dataframe(results_df.style.background_gradient(subset=['AI Score'], cmap='RdYlGn'), use_container_width=True)
        else:
            st.warning("No stocks passed the filters. Try lowering the Volume or Volatility requirements.")

# --- PAGE: WAR ROOM (Modified for Summary) ---
def show_war_room():
    st.title("🦅 Tactical War Room")
    # Existing Watchlist logic here...
    st.info("Top recommendations from your watchlist and global scans appear here.")

# --- NAVIGATION ---
pg = st.navigation([
    st.Page(show_war_room, title="War Room", icon="🦅"),
    st.Page(show_global_discovery, title="Global Discovery", icon="🚀"),
])
pg.run()