import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
import pytz
from datetime import datetime
import requests
import base64

# --- SETTINGS & CONFIG ---
st.set_page_config(page_title="Trade AI: War Room", layout="wide")

WATCHLIST = [
    {"Ticker": "CBRE",   "Market": "US"},
    {"Ticker": "SAN",    "Market": "US"}, # Alpha Vantage works best with US tickers or specific suffixes
    {"Ticker": "CSCO",   "Market": "US"},
]

# --- ALPHA VANTAGE FETCHING ---
def fetch_processed_data(ticker):
    """Fetches data from Alpha Vantage and calculates indicators."""
    api_key = st.secrets.get("ALPHA_VANTAGE_KEY")
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={api_key}'
    
    try:
        r = requests.get(url)
        data = r.json()
        
        # Handle API Limit/Error
        if "Time Series (Daily)" not in data:
            st.error(f"Alpha Vantage Error for {ticker}: {data.get('Note', 'Check API Key')}")
            return pd.DataFrame()

        # Parse JSON to DataFrame
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)

        # Basic Indicators (Replacing your DataFetcher logic for stability)
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Volatility'] = df['Close'].pct_change().rolling(window=21).std()
        
        return df.dropna()
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# --- SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.title("🦅 Dip-Hunter Access")
        if not st.secrets.get("password"):
            st.error("⚠️ No password found in secrets.toml.")
        pwd_input = st.text_input("Enter credentials", type="password")
        if st.button("Login"):
            if pwd_input == st.secrets.get("password"):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Password incorrect")
        return False
    return True

# --- TRADING 212 INTEGRATION ---
def get_t212_positions():
    key_id = st.secrets.get("t212_key_id", "").strip()
    secret = st.secrets.get("t212_secret", "").strip()
    if not key_id or not secret: return None

    auth_bytes = f"{key_id}:{secret}".encode("utf-8")
    encoded_auth = base64.b64encode(auth_bytes).decode("utf-8")
    headers = {"Authorization": f"Basic {encoded_auth}", "Accept": "application/json"}
    
    try:
        res = requests.get("https://live.trading212.com/api/v0/equity/positions", headers=headers)
        return res.json() if res.status_code == 200 else None
    except: return None

# --- SHARED UTILITIES ---
@st.cache_resource
def load_assets():
    try: return joblib.load('models/random_forest_model.pkl')
    except: return None

ml_model = load_assets()

def create_price_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='#00d4ff', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50 MA', line=dict(color='rgba(255, 255, 255, 0.4)', dash='dash')))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# --- PAGE: WAR ROOM ---
def show_war_room():
    st.title("🦅 Trade AI: War Room")
    st.markdown("---")
    
    dashboard_data = []
    with st.spinner('Fetching Alpha Vantage Data...'):
        for item in WATCHLIST:
            df = fetch_processed_data(item["Ticker"])
            if not df.empty:
                latest = df.iloc[-1]
                # ML Logic
                features = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                prob = ml_model.predict_proba(features)[0][1] if ml_model else 0.5
                dashboard_data.append({"Ticker": item["Ticker"], "Price": latest['Close'], "RSI": latest['RSI'], "Conviction": prob})

    if dashboard_data:
        st.table(pd.DataFrame(dashboard_data))

# --- PAGE: PORTFOLIO ---
def show_portfolio():
    st.title("💰 My Live Portfolio")
    positions = get_t212_positions()
    
    if positions:
        portfolio_list = []
        for pos in positions:
            instrument = pos.get('instrument', {})
            ticker = str(instrument.get('ticker', 'UNKNOWN')).split('_')[0] 
            wallet = pos.get('walletImpact', {})
            
            portfolio_list.append({
                "Ticker": ticker,
                "Qty": pos.get('quantity', 0), 
                "Price": pos.get('currentPrice', 0), 
                "P&L": f"{wallet.get('unrealizedProfitLoss', 0.0):.2f} {wallet.get('currency', '')}"
            })
        
        df_p = pd.DataFrame(portfolio_list)
        st.dataframe(df_p, use_container_width=True)

        if st.button("Run AI Health Check"):
            for t in df_p['Ticker']:
                with st.expander(f"Audit: {t}"):
                    data = fetch_processed_data(t)
                    if not data.empty:
                        latest = data.iloc[-1]
                        feat = [[latest['RSI'], latest['MA50'], latest['MA200'], latest['Volatility'], latest['Volume']]]
                        prob = ml_model.predict_proba(feat)[0][1] if ml_model else 0.5
                        st.metric("AI Conviction", f"{prob:.1%}")

# --- NAVIGATION ---
if check_password():
    pg = st.navigation([
        st.Page(show_war_room, title="War Room", icon="🦅"),
        st.Page(show_portfolio, title="My Portfolio", icon="💰")
    ])
    pg.run()