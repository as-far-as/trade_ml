import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go
from etl.fetcher import DataFetcher, get_index_tickers
from etl.indicators import add_indicators
from etl.sentiment_fetcher import SocialSentimentFetcher
from data.llm_agent import SentimentAnalyzer
import requests
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Trade Pro", layout="wide")

# --- INITIALIZE AGENTS ---
sentiment_fetcher = SocialSentimentFetcher()
sentiment_analyzer = SentimentAnalyzer()

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    .top-pick-card { border: 1px solid #3d4455; padding: 20px; border-radius: 10px; margin-bottom: 10px; min-height: 180px; }
    .buy-signal { color: #00ff00; font-weight: bold; }
    .sell-signal { color: #ff4b4b; font-weight: bold; }
    .hold-signal { color: #ffa500; font-weight: bold; }
    .stButton>button { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load('models/random_forest_model.pkl')
    except:
        return None

model = load_ml_model()

# --- UTILS ---
def get_prediction_score(df, sentiment=0.0):
    if model is None: return 0.5001
    if df.empty or len(df) < 200: return 0.5002
    try:
        processed = add_indicators(df)
        if processed.empty: return 0.5003
        latest = processed.iloc[-1]
        # features = ['RSI', 'MA50', 'MA200', 'Volatility', 'Relative_Volume', 'Sentiment']
        X = np.array([[
            latest['RSI'], 
            latest['MA50'], 
            latest['MA200'], 
            latest['Volatility'], 
            latest['Relative_Volume'],
            sentiment
        ]])
        return float(model.predict_proba(X)[0][1])
    except:
        return 0.5004

@st.cache_data(ttl=3600)
def fetch_top_picks(index_name, min_vol=0, min_vty=0):
    tickers = get_index_tickers(index_name)
    if not tickers: return pd.DataFrame()
    
    with st.status(f"Downloading data for {len(tickers)} stocks...", expanded=False) as status:
        data = yf.download(tickers, period="2y", progress=False, group_by='ticker')
        status.update(label="Analysis complete!", state="complete")
        
    results = []
    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.levels[0]: continue
                t_df = data[ticker].dropna()
            else:
                t_df = data.dropna()
                
            if len(t_df) < 200: continue
            avg_vol = (t_df['Close'] * t_df['Volume']).tail(20).mean()
            vty = t_df['Close'].pct_change().tail(21).std()
            
            if avg_vol < min_vol or vty < min_vty: continue
            
            score = get_prediction_score(t_df, sentiment=0.0)
            results.append({
                "Ticker": ticker, 
                "Score": score, 
                "Price": round(float(t_df['Close'].iloc[-1]), 2),
                "Volat": round(float(vty), 4), 
                "Vol($M)": round(float(avg_vol/1e6), 2)
            })
        except: continue
        
    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values(by="Score", ascending=False)

def get_recommendation(score):
    if score > 0.60: return "Strong Buy", "buy-signal"
    if score > 0.52: return "Buy", "buy-signal"
    if score < 0.40: return "Strong Sell", "sell-signal"
    if score < 0.48: return "Sell", "sell-signal"
    return "Hold", "hold-signal"

def get_t212_portfolio():
    key = st.secrets.get("t212_key_id")
    if not key: return []
    try:
        res = requests.get("https://live.trading212.com/api/v0/equity/portfolio", headers={"Authorization": key})
        return res.json() if res.status_code == 200 else []
    except: return []

# --- NAVIGATION ---
st.sidebar.title("🚀 Trade AI Pro")
page = st.sidebar.radio("Navigation", ["Global Top Picks", "Specific Stock Analysis", "My Portfolio (T212)"])

# --- PAGE: DASHBOARD ---
if page == "Global Top Picks":
    st.title("🌎 Global Market Discovery")
    st.write("Automatically scan the S&P 500, FTSE 100, and IBEX 35 for the best AI-rated opportunities.")
    
    with st.expander("Scanner Advanced Filters"):
        c1, c2 = st.columns(2)
        m_vol = c1.number_input("Min Daily Liquidity ($M)", value=1.0) * 1e6
        m_vty = c2.slider("Min Volatility (21-day std)", 0.0, 0.05, 0.01, 0.001)
    
    t1, t2, t3 = st.tabs(["🇺🇸 USA (S&P 500)", "🇬🇧 UK (FTSE 100)", "🇪🇸 Spain (IBEX 35)"])
    
    for i, (tab, idx) in enumerate(zip([t1, t2, t3], ["SP500", "FTSE100", "IBEX35"])):
        with tab:
            if st.button(f"🚀 Start AI Scan: {idx}", key=f"btn_{idx}"):
                res = fetch_top_picks(idx, m_vol, m_vty)
                st.session_state[f"res_{idx}"] = res
            
            if f"res_{idx}" in st.session_state:
                res = st.session_state[f"res_{idx}"]
                if not res.empty:
                    st.subheader(f"Top 10 Picks for {idx}")
                    cols = st.columns(5)
                    for j, (_, row) in enumerate(res.head(10).iterrows()):
                        rec, css = get_recommendation(row['Score'])
                        with cols[j % 5]:
                            st.markdown(f"""
                                <div class='top-pick-card'>
                                    <h4>{row['Ticker']}</h4>
                                    <p>Price: <b>{row['Price']}</b></p>
                                    <p>Score: <b>{row['Score']:.4f}</b></p>
                                    <p class='{css}'>{rec}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.divider()
                    st.subheader("All Results")
                    st.dataframe(res.style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)
                else:
                    st.warning("No stocks matched the current filters. Try lowering the Minimum Liquidity or Volatility.")

# --- PAGE: SEARCH ---
elif page == "Specific Stock Analysis":
    st.title("🔍 Individual Stock Intelligence")
    st.write("Analyze any stock ticker from Yahoo Finance and get a real-time AI score and recommendation.")
    
    ticker = st.text_input("Enter Ticker (e.g. TSLA, NVDA, BARC.L, ITX.MC)", "").upper()
    
    if ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            hist = yf.download(ticker, period="2y", progress=False)
            if not hist.empty and len(hist) >= 200:
                # NEW: Fetch and analyze sentiment
                sentiment_score = 0.0
                with st.expander("Sentiment & Context Analysis", expanded=True):
                    context = sentiment_fetcher.get_all_context(ticker)
                    if context:
                        st.info(f"Found {len(context)} recent mentions on Reddit and News.")
                        sentiment_score = sentiment_analyzer.analyze_sentiment(context)
                        st.metric("Contextual Sentiment Score", f"{sentiment_score:.2f}", 
                                  delta="Bullish" if sentiment_score > 0 else "Bearish" if sentiment_score < 0 else "Neutral")
                    else:
                        st.warning("No recent social or news context found for this ticker.")

                score = get_prediction_score(hist, sentiment=sentiment_score)
                rec, css = get_recommendation(score)
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = go.Figure(data=[go.Candlestick(x=hist.index, 
                        open=hist['Open'], high=hist['High'], 
                        low=hist['Low'], close=hist['Close'])])
                    fig.update_layout(title=f"{ticker} Technical Chart", template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    st.markdown(f"""
                        <div style='background-color: #1e2130; padding: 20px; border-radius: 10px; border: 1px solid #3d4455;'>
                            <h2 style='margin-top:0;'>AI Verdict</h2>
                            <p style='font-size: 12px; color: #888;'>Includes Technical + Sentiment Data</p>
                            <h1 style='color: {"#00ff00" if "Buy" in rec else "#ff4b4b" if "Sell" in rec else "#ffa500"}; font-size: 48px;'>{score:.4f}</h1>
                            <h3 class='{css}'>{rec}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    processed = add_indicators(hist)
                    if not processed.empty:
                        latest = processed.iloc[-1]
                        st.divider()
                        st.subheader("Key Indicators")
                        st.metric("RSI (14)", f"{latest['RSI']:.2f}", delta_color="off")
                        st.metric("MA 50/200", f"{latest['MA50']:.1f} / {latest['MA200']:.1f}")
                        st.metric("Volatility", f"{latest['Volatility']:.4f}")
                        st.metric("Relative Volume", f"{latest['Relative_Volume']:.2f}")
            else:
                st.error("Ticker not found or insufficient historical data (need at least 200 trading days).")

# --- PAGE: PORTFOLIO ---
elif page == "My Portfolio (T212)":
    st.title("💼 Live Trading 212 Portfolio")
    st.write("Connected to your live portfolio. Run AI analysis on your current holdings.")
    
    pos = get_t212_portfolio()
    if pos:
        rows = []
        for p in pos:
            t = p['ticker'].split('_')[0]
            # Mapping T212 format to Yahoo
            if "_" in p['ticker']:
                s = p['ticker'].split('_')[1]
                if s == "L": t += ".L"
                elif s == "EQ" and "." not in t: pass # US
            rows.append({"Ticker": t, "Qty": p['quantity'], "Avg Price": p['averagePrice'], "Current Price": p['currentPrice'], "P&L": p['ppl']})
        
        df = pd.DataFrame(rows)
        st.dataframe(df.style.background_gradient(subset=['P&L'], cmap='RdYlGn'), use_container_width=True)
        
        if st.button("🔍 Run AI Deep Scan on Portfolio"):
            with st.spinner("Processing holdings..."):
                scores = []
                recs = []
                for ticker in df['Ticker']:
                    try:
                        h = yf.download(ticker, period="2y", progress=False)
                        # Default sentiment to 0 for portfolio scan to save time
                        s = get_prediction_score(h, sentiment=0.0)
                        r, _ = get_recommendation(s)
                        scores.append(s)
                        recs.append(r)
                    except:
                        scores.append(0.5)
                        recs.append("Error")
                df['AI Score'] = scores
                df['AI Rec'] = recs
                st.subheader("Portfolio AI Analysis")
                st.dataframe(df.sort_values("AI Score", ascending=False).style.background_gradient(subset=['AI Score'], cmap='RdYlGn'), use_container_width=True)
    else:
        st.info("No positions found or API key missing in st.secrets.")
