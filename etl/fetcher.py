import pandas as pd
import requests
import streamlit as st

@st.cache_data(ttl=86400)
def get_index_tickers(index_name="SP500"):
    """Fetches stock lists from Wikipedia with dynamic column detection."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    # Current Wikipedia table indices for 2026
    config = {
        "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0),
        "FTSE100": ("https://en.wikipedia.org/wiki/FTSE_100_Index", 4),
        "IBEX35": ("https://en.wikipedia.org/wiki/IBEX_35", 1)
    }
    
    try:
        url, table_idx = config[index_name]
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text, flavor='lxml')
        df = tables[table_idx]

        # DYNAMIC COLUMN SEARCH: Prevents 'EPIC' or 'Ticker' KeyErrors
        # Looks for common variations of ticker column names
        possible_cols = ['Symbol', 'Ticker', 'EPIC', 'Ticker symbol', 'Code', 'Ticker.MC']
        ticker_col = next((c for c in df.columns if c in possible_cols), None)

        if not ticker_col:
            st.error(f"Could not find ticker column for {index_name}. Found: {list(df.columns)}")
            return []

        tickers = df[ticker_col].astype(str).str.strip().tolist()

        # APPLY REGIONAL SUFFIXES
        if index_name == "FTSE100":
            return [f"{t}.L" if not t.endswith('.L') else t for t in tickers]
        if index_name == "IBEX35":
            # IBEX tickers often already have .MC on Wikipedia
            return [f"{t}.MC" if not t.endswith('.MC') else t for t in tickers]
        
        # Clean S&P 500 symbols (e.g., BRK.B to BRK-B for Yahoo)
        return [t.replace('.', '-') for t in tickers]

    except Exception as e:
        st.error(f"Scraper failed for {index_name}: {e}")
        return []

class DataFetcher:
    def add_indicators(self, df):
        """Standardizes technical indicators for the ML model."""
        if df.empty or len(df) < 20:
            return pd.DataFrame()
        
        df = df.copy()
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Trend and Volatility
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(window=21).std()
        
        return df.dropna()