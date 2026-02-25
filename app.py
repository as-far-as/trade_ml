import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import requests
from etl.fetcher import DataFetcher, get_index_tickers

# --- INITIALIZATION ---
st.set_page_config(page_title="Trade AI Pro", layout="wide")
fetcher = DataFetcher()

@st.cache_resource
def load_model():
    try:
        return joblib.load('models/random_forest_model.pkl')
    except:
        return None

model = load_model()

# --- T212 API RESTORED ---
def get_t212_data():
    api_key = st.secrets.get("T212_API_KEY")
    if not api_key:
        st.warning("T212_API_KEY not found in secrets.")
        return []
    
    url = "https://live.trading212.com/api/v0/equity/portfolio"
    headers = {"Authorization": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        return response.json() if response.status_code == 200 else []
    except:
        return []

def get_ai_score(df):
    if model is None or df.empty: return 0.5
    try:
        processed = fetcher.add_indicators(df)
        if processed.empty: return 0.5
        latest = processed.iloc[-1]
        features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
        return model.predict_proba(features)[0][1]
    except:
        return 0.5

# --- PAGE: GLOBAL DISCOVERY ---
def show_discovery():
    st.title("🚀 Global AI Discovery")
    
    with st.sidebar:
        idx_choice = st.selectbox("Index", ["SP500", "FTSE100", "IBEX35"])
        min_vol = st.number_input("Min Vol ($M)", value=0.1) * 1_000_000
        min_vty = st.slider("Min Volatility %", 0.0, 5.0, 0.1) / 100

    if st.button("Start AI Scan"):
        tickers = get_index_tickers(idx_choice)
        if not tickers: return

        with st.spinner(f"Scanning {len(tickers)} stocks..."):
            # Use group_by='column' to keep Tickers as columns for easier filtering
            data = yf.download(tickers, period="6mo", interval="1d", group_by='column', threads=True)
            
            results = []
            for t in tickers:
                try:
                    # Extract OHLCV for specific ticker
                    ticker_df = pd.DataFrame({
                        'Close': data['Close'][t],
                        'Volume': data['Volume'][t]
                    }).dropna()
                    
                    if len(ticker_df) < 50: continue
                    
                    # Calculation
                    avg_liq = (ticker_df['Close'] * ticker_df['Volume']).tail(5).mean()
                    vty = ticker_df['Close'].pct_change().tail(21).std()

                    if avg_liq >= min_vol and vty >= min_vty:
                        score = get_ai_score(ticker_df)
                        results.append({
                            "Ticker": t,
                            "AI Score": score,
                            "Price": round(ticker_df['Close'].iloc[-1], 2),
                            "Volat%": f"{vty:.2%}",
                            "Liq($M)": round(avg_liq / 1_000_000, 2)
                        })
                except: continue

            if results:
                df_res = pd.DataFrame(results).sort_values("AI Score", ascending=False)
                st.dataframe(df_res.style.background_gradient(subset=['AI Score'], cmap='RdYlGn'))
            else:
                st.warning("No matches. Try lowering filters.")

# --- PAGE: TRADING 212 ---
def show_t212():
    st.title("💰 Trading 212 Portfolio")
    if st.button("Sync Live Portfolio"):
        positions = get_t212_data()
        if not positions:
            st.error("Could not fetch API data.")
            return

        final_rows = []
        for pos in positions:
            t = pos['ticker'].split('_')[0]
            hist = yf.download(t, period="6mo", progress=False)
            score = get_ai_score(hist)
            final_rows.append({
                "Ticker": t,
                "Quantity": pos['quantity'],
                "AI Score": score,
                "P&L": pos.get('ppl', 0)
            })
        
        st.table(pd.DataFrame(final_rows).sort_values("AI Score", ascending=False).style.background_gradient(subset=['AI Score'], cmap='RdYlGn'))

# --- NAV ---
pg = st.navigation([
    st.Page(show_discovery, title="Global Discovery", icon="🚀"),
    st.Page(show_t212, title="Trading 212", icon="💰")
])
pg.run()